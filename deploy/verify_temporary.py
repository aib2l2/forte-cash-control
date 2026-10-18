"""Test public routing with synthetic video only; never output credentials or frames."""

import asyncio
import base64
import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import cv2
import httpx
import numpy as np
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "storage/public_demo"


async def check_socket(url, authorization):
    uri = url.replace("https://", "wss://") + "/_stcore/stream"
    async with connect(
        uri,
        additional_headers={"Authorization": authorization},
        origin=url,
        subprotocols=["streamlit"],
        open_timeout=20,
        proxy=None,
        ssl=ssl.create_default_context(cafile=certifi.where()),
    ) as ws:
        from streamlit.proto.BackMsg_pb2 import BackMsg
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

        back = BackMsg()
        back.rerun_script.query_string = ""
        ws_msg = back.SerializeToString()
        await ws.send(ws_msg)
        kinds, text = [], []
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            message = ForwardMsg()
            message.ParseFromString(await asyncio.wait_for(ws.recv(), timeout=20))
            kind = message.WhichOneof("type")
            kinds.append(kind)
            if kind == "delta" and message.delta.HasField("new_element"):
                element = message.delta.new_element
                if element.HasField("markdown"):
                    text.append(element.markdown.body)
                if element.HasField("alert"):
                    text.append(element.alert.body)
                if element.HasField("exception"):
                    raise RuntimeError("Public Streamlit app emitted an exception")
            if kind == "script_finished":
                break
        joined = " ".join(text)
        if "ML-сервер недоступен" in joined or "Готова к анализу" not in joined:
            raise RuntimeError("UI did not confirm ready model")
        return {
            "websocket_connected": True,
            "script_finished": "script_finished" in kinds,
            "model_ready_displayed": True,
            "messages": len(kinds),
        }


def main():
    config = json.loads((STATE / "private_config.json").read_text())
    url = config["url"]
    headers = {"X-API-Key": config["api_key"]}
    auth = (config["username"], config["password"])
    checks = {}
    with httpx.Client(trust_env=False, timeout=25) as client:
        checks["ui_denies_anonymous"] = client.get(url + "/").status_code == 401
        checks["ui_denies_wrong_password"] = (
            client.get(url + "/", auth=("forte", "invalid")).status_code == 401
        )
        checks["api_denies_anonymous"] = client.get(url + "/api/v1/model").status_code == 401
        checks["api_denies_wrong_key"] = (
            client.get(url + "/api/v1/model", headers={"X-API-Key": "invalid"}).status_code == 401
        )
        r = client.get(url + "/", auth=auth)
        checks["ui_html_loaded"] = r.status_code == 200 and "streamlit" in r.text.lower()
        checks["no_cache"] = "no-store" in {
            value.strip().lower() for value in r.headers.get("cache-control", "").split(",")
        }
        checks["health"] = client.get(url + "/health").status_code == 200
        r = client.get(url + "/api/v1/model", headers=headers)
        r.raise_for_status()
        model = r.json()
        checks["active_v2"] = model["ready"] and model["active_version"] == "typed_v2"
        allowed = set(model["allowed_types"])
        checks["reject_non_mov"] = (
            client.post(
                url + "/api/v1/analyze",
                headers=headers,
                files={"file": ("wrong.txt", b"not a video", "text/plain")},
            ).status_code
            == 415
        )
        checks["reject_empty"] = (
            client.post(
                url + "/api/v1/analyze",
                headers=headers,
                files={"file": ("empty.mov", b"", "video/quicktime")},
            ).status_code
            == 400
        )
        checks["unknown_job"] = (
            client.get(url + "/api/v1/jobs/nonexistent", headers=headers).status_code == 404
        )
    auth_header = "Basic " + base64.b64encode((auth[0] + ":" + auth[1]).encode()).decode()
    websocket = asyncio.run(check_socket(url, auth_header))
    path = STATE / "synthetic_probe.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (160, 240))
    if not writer.isOpened():
        raise RuntimeError("Synthetic encoder unavailable")
    for index in range(60):
        frame = np.full((240, 160, 3), 20 + index * 3, np.uint8)
        cv2.rectangle(frame, (20, 20), (80, 80), (40, 100, 160), -1)
        writer.write(frame)
    writer.release()

    def upload(_):
        start = time.monotonic()
        with httpx.Client(trust_env=False, timeout=45) as client, path.open("rb") as file:
            r = client.post(
                url + "/api/v1/analyze",
                headers=headers,
                files={"file": (path.name, file, "video/quicktime")},
            )
            if r.status_code != 202:
                raise RuntimeError("Public upload failed")
            job_id = r.json()["job_id"]
            while time.monotonic() - start < 300:
                result = client.get(url + "/api/v1/jobs/" + job_id, headers=headers)
                result.raise_for_status()
                payload = result.json()
                if payload["status"] != "processing":
                    break
                time.sleep(0.5)
            if payload["status"] != "done":
                raise RuntimeError("Public video processing failed")
            if payload["verdict"] not in {"violation", "no_violation"}:
                raise RuntimeError("Invalid verdict")
            frames_checked = 0
            capture = cv2.VideoCapture(str(path))
            for event in payload["violations"]:
                assert event["type"] in allowed and 0 <= event["timecode_sec"] < 4.0
                assert 1 <= len(event["evidence_frames"]) <= 3
                for frame in event["evidence_frames"]:
                    stamp = frame["timestamp_sec"]
                    assert 0 <= stamp < 4.0
                    image = cv2.imdecode(
                        np.frombuffer(
                            base64.b64decode(frame["image_base64"], validate=True), np.uint8
                        ),
                        1,
                    )
                    assert image is not None
                    capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
                    ok, source = capture.read()
                    assert ok and image.shape == source.shape
                    assert float(np.mean(abs(image.astype(float) - source.astype(float)))) < 4.0
                    frames_checked += 1
            capture.release()
            return {
                "status": payload["status"],
                "seconds_including_upload_queue_poll": round(time.monotonic() - start, 2),
                "events": len(payload["violations"]),
                "real_source_frames_verified": frames_checked,
            }

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(upload, range(2)))
    checks["two_uploads_done_under_300s"] = all(
        r["seconds_including_upload_queue_poll"] < 300 for r in jobs
    )
    checks["websocket_and_render"] = (
        websocket["script_finished"] and websocket["model_ready_displayed"]
    )
    report = {
        "url": url,
        "temporary": True,
        "model": "typed_v2",
        "checks": checks,
        "websocket": websocket,
        "jobs": jobs,
        "private_dataset_uploaded": False,
        "test_video": "synthetic, not an accuracy assessment",
        "passed": all(checks.values()),
    }
    (STATE / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["passed"]


if __name__ == "__main__":
    main()
