"""Run: python scripts/check_api.py /path/to/video.mov"""

import argparse
import json
import time
from pathlib import Path

import httpx

from backend.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузить .mov и дождаться ответа API.")
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    started = time.monotonic()
    with httpx.Client(
        base_url=settings.api_base_url,
        headers={"X-API-Key": settings.api_key},
        timeout=60,
        trust_env=False,
    ) as client:
        with args.video.open("rb") as video:
            response = client.post(
                "/api/v1/analyze",
                files={"file": (args.video.name, video, "video/quicktime")},
            )
        response.raise_for_status()
        job_id = response.json()["job_id"]
        print(f"Задача принята: {job_id}")
        while time.monotonic() - started < 300:
            response = client.get(f"/api/v1/jobs/{job_id}", timeout=10)
            response.raise_for_status()
            result = response.json()
            if result["status"] != "processing":
                # Avoid printing potentially large/private evidence images to the terminal.
                for event in result.get("violations", []):
                    for frame in event["evidence_frames"]:
                        frame["image_base64"] = "<JPEG base64 скрыт>"
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"Время: {time.monotonic() - started:.1f} с")
                return 0 if result["status"] == "done" else 1
            time.sleep(1)
    print("Превышен лимит 5 минут. Проверьте производительность и очередь.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
