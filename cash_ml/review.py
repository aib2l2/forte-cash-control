"""Append-only human annotations, separate from immutable Excel point labels."""

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cash_ml.features import sha256


def review_tasks(records):
    tasks = []
    for record in records:
        if record["split"] != "train":
            continue
        events = record["annotations"] or [{"type": None, "timecode": None}]
        for index, event in enumerate(events):
            tasks.append({
                "task_id": f"{record['sha256']}:{index}", "video": record["id"],
                "source_sha256": record["sha256"], "aliases": record["aliases"],
                "split": "train", "original_type": event["type"],
                "original_point_sec": event["timecode"], "start_sec": None,
                "end_sec": None, "observable_actions": "", "uncertainties": "",
                "review_status": "pending", "path": record["path"],
            })
    return tasks


def append_review(destination, task, review, allowed_types, duration):
    if task["split"] != "train":
        raise ValueError("This annotation workspace is limited to training data")
    if sha256(Path(task["path"])) != task["source_sha256"]:
        raise ValueError("Исходное видео изменилось")
    if not review.get("reviewer", "").strip() or not review.get("locally_viewed"):
        raise ValueError("Нужны имя проверяющего и подтверждение локального просмотра")
    status = review["review_status"]
    if status not in {"confirmed_event", "confirmed_normal_interval", "unresolved"}:
        raise ValueError("Неизвестный статус проверки")
    event_type = review.get("type")
    if event_type is not None and event_type not in allowed_types:
        raise ValueError("Тип отсутствует в актуальном словаре")
    if status == "confirmed_event" and event_type is None:
        raise ValueError("Для события нужен тип из Excel")
    start, end, point = [review.get(k) for k in ("start_sec", "end_sec", "point_sec")]
    for t in (start, end, point):
        if t is not None and (not math.isfinite(t) or not 0 <= t <= duration):
            raise ValueError("Время вне границ ролика")
    if (start is None) != (end is None) or (start is not None and not start < end):
        raise ValueError("Границы интервала должны быть полными и возрастающими")
    if start is not None and point is not None:
        raise ValueError("Выберите интервал или точечную метку")
    if status == "confirmed_normal_interval":
        if start is None or event_type is not None:
            raise ValueError("Нормальный пример требует явно просмотренного интервала без типа нарушения")
    if status == "confirmed_event" and start is None and point is None:
        raise ValueError("Нужны подтверждённый интервал или точечная метка")
    if status != "unresolved" and not review.get("observable_actions", "").strip():
        raise ValueError("Опишите видимые действия без догадок о намерениях или полномочиях")
    row = {
        **review, "task_id": task["task_id"], "source_sha256": task["source_sha256"],
        "video": task["video"], "split": "train", "original_type": task["original_type"],
        "original_point_sec": task["original_point_sec"], "revision_id": uuid4().hex,
        "reviewed_at": datetime.now(UTC).isoformat(), "used_for_training": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, content) != len(content):
            raise OSError("Incomplete annotation write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return row["revision_id"]
