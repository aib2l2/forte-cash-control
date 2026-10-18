"""Password-protected temporary demo on this Mac; no dataset or review UI exposed."""

import argparse
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "storage/public_demo"
CONFIG = STATE / "private_config.json"
PIDS = STATE / "processes.json"
PYTHON = ROOT / ".venv/bin/python"


def private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(text)
    path.chmod(0o600)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def spawn(name, command, env, state):
    log = STATE / (name + ".log")
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "w") as output:
        proc = subprocess.Popen(
            [str(x) for x in command],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state[name] = {"pid": proc.pid, "command": [str(x) for x in command]}
    private_write(PIDS, json.dumps(state, indent=2))
    return proc


def wait_http(url, headers=None, timeout=25):
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=2) as client:
        while time.monotonic() < deadline:
            try:
                r = client.get(url, headers=headers)
                if r.status_code == 200:
                    return r
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise RuntimeError("Local demo service did not become healthy")


def start():
    STATE.mkdir(parents=True, exist_ok=True)
    STATE.chmod(0o700)
    if PIDS.exists() and any(
        process_alive(v["pid"]) for v in json.loads(PIDS.read_text()).values()
    ):
        raise RuntimeError("Demo is already running; use status or stop")
    for port in (8010, 8510, 8765):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is already in use") from exc
    caddy, tunnel = STATE / "bin/caddy", STATE / "bin/cloudflared"
    if not caddy.is_file() or not tunnel.is_file():
        raise RuntimeError(
            "Verified Caddy and cloudflared binaries are required in storage/public_demo/bin"
        )
    config = {
        "username": "forte",
        "password": secrets.token_urlsafe(18),
        "api_key": secrets.token_urlsafe(32),
        "cookie_secret": secrets.token_urlsafe(32),
    }
    private_write(CONFIG, json.dumps(config, indent=2))
    hashed = subprocess.run(
        [str(caddy), "hash-password"],
        input=config["password"] + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    caddyfile = """{
    admin off
    auto_https off
}
:8765 {
    bind 127.0.0.1
    header {
        Cache-Control "no-store"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "no-referrer"
    }
    request_body {
        max_size 101MB
    }
    handle /api/v1/* {
        reverse_proxy 127.0.0.1:8010
    }
    handle /health {
        reverse_proxy 127.0.0.1:8010
    }
    handle {
        basic_auth {
            forte PASSWORD_HASH
        }
        reverse_proxy 127.0.0.1:8510
    }
}
""".replace("PASSWORD_HASH", hashed)
    private_write(STATE / "Caddyfile", caddyfile)
    validation = subprocess.run(
        [str(caddy), "validate", "--config", str(STATE / "Caddyfile"), "--adapter", "caddyfile"],
        capture_output=True,
        text=True,
    )
    if validation.returncode:
        raise RuntimeError("Caddy configuration validation failed")
    env = dict(
        os.environ,
        API_KEY=config["api_key"],
        API_BASE_URL="http://127.0.0.1:8010",
        STORAGE_DIR=str(STATE / "results"),
        MAX_PENDING_JOBS="4",
        JOB_DEADLINE_SECONDS="290",
        STREAMLIT_SERVER_COOKIE_SECRET=config["cookie_secret"],
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
    )
    state = {}
    spawn(
        "api",
        [
            PYTHON,
            "-m",
            "uvicorn",
            "cash_ml.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
            "--workers",
            "1",
        ],
        env,
        state,
    )
    wait_http("http://127.0.0.1:8010/api/v1/model", {"X-API-Key": config["api_key"]})
    spawn(
        "gateway",
        [caddy, "run", "--config", STATE / "Caddyfile", "--adapter", "caddyfile"],
        env,
        state,
    )
    wait_http("http://127.0.0.1:8765/health")
    with httpx.Client(trust_env=False) as client:
        assert client.get("http://127.0.0.1:8765/").status_code == 401
        assert client.get("http://127.0.0.1:8765/api/v1/model").status_code == 401
    private_write(STATE / "tunnel.log", "")
    proc = spawn(
        "tunnel",
        [
            tunnel,
            "tunnel",
            "--no-autoupdate",
            "--protocol",
            "http2",
            "--url",
            "http://127.0.0.1:8765",
        ],
        env,
        state,
    )
    deadline, url = time.monotonic() + 50, None
    while time.monotonic() < deadline:
        log = (STATE / "tunnel.log").read_text()
        matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
        if matches:
            url = matches[0]
            break
        if proc.poll() is not None:
            raise RuntimeError("Tunnel process exited; inspect local tunnel.log")
        time.sleep(0.5)
    if not url:
        raise RuntimeError("Tunnel URL was not assigned; local protected gateway is ready")
    env["STREAMLIT_BROWSER_SERVER_ADDRESS"] = url.removeprefix("https://")
    env["STREAMLIT_BROWSER_SERVER_PORT"] = "443"
    spawn(
        "ui",
        [
            PYTHON,
            "-m",
            "streamlit",
            "run",
            "cash_ml/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8510",
            "--browser.gatherUsageStats",
            "false",
        ],
        env,
        state,
    )
    wait_http("http://127.0.0.1:8510/_stcore/health")
    config["url"] = url
    private_write(CONFIG, json.dumps(config, indent=2))
    private_write(
        STATE / "access.txt",
        "ВРЕМЕННОЕ ДЕМО — работает, пока Mac включён и доступен в сети.\n"
        + "URL: "
        + url
        + "\nЛогин: "
        + config["username"]
        + "\nПароль: "
        + config["password"]
        + "\nAPI X-API-Key: "
        + config["api_key"]
        + "\n"
        + "Не публикуйте этот файл. Остановка: .venv/bin/python deploy/temporary_public.py stop\n",
    )
    print(
        json.dumps(
            {
                "url": url,
                "credentials_file": str(STATE / "access.txt"),
                "temporary": True,
                "model": "typed_v2",
            }
        )
    )


def stop():
    if not PIDS.exists():
        print("No demo processes recorded")
        return
    state = json.loads(PIDS.read_text())
    for name in ("tunnel", "gateway", "ui", "api"):
        item = state.get(name)
        if item and process_alive(item["pid"]):
            # Verify identity before stopping a possibly reused PID.
            command = subprocess.run(
                ["ps", "-p", str(item["pid"]), "-o", "command="], capture_output=True, text=True
            ).stdout
            marker = {"tunnel": "cloudflared", "gateway": "caddy run", "ui": "8510", "api": "8010"}[
                name
            ]
            if marker not in command:
                raise RuntimeError("Process identity changed; not stopping an unrelated process")
            os.killpg(item["pid"], signal.SIGTERM)
    private_write(PIDS, "{}")
    print("Temporary demo stopped; original localhost application unchanged")


def status():
    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    state = json.loads(PIDS.read_text()) if PIDS.exists() else {}
    print(
        json.dumps(
            {
                "url": config.get("url"),
                "processes": {k: process_alive(v["pid"]) for k, v in state.items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    {"start": start, "stop": stop, "status": status}[args.action]()
