"""Run separately on 127.0.0.1:8502. Never expose this annotation app publicly."""

import json
import os
from pathlib import Path

import cv2
import streamlit as st

from backend.config import PROJECT_ROOT
from cash_ml.context_features import letterbox
from cash_ml.review import append_review, review_tasks
from cash_ml.rules import registry

st.set_page_config(page_title="ForteAI · локальная разметка", layout="wide")
st.title("Локальная проверка разметки")
st.caption("Только обучающая часть. Видео и заметки остаются на этом компьютере.")
st.info("Просмотрите эпизод и контекст. Если начало и конец не видны, сохраните точку. "
        "Сохранение создаёт отдельную запись; исходный Excel и модель не меняются.")
run = Path(os.getenv("REVIEW_RUN", (PROJECT_ROOT / "storage/runtime/temporal_run.txt").read_text().strip()))
records = json.loads((run / "previous_grouped_records.json").read_text())
tasks = review_tasks(records)
allowed = json.loads((run / "protocol.json").read_text())["allowed_types"]
destination = PROJECT_ROOT / "storage/annotations/reviews.jsonl"
task = st.selectbox("Эпизод", tasks, format_func=lambda x: (
    f"{x['video']} · {x['original_type'] or 'нормальный пример'} · {x['original_point_sec']} с"
))
source = Path(task["path"])
capture = cv2.VideoCapture(str(source))
duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1)
capture.release()
st.video(str(source), start_time=max(0.0, float(task["original_point_sec"] or 0) - 4), muted=True)
st.caption(f"Длительность {duration:.2f} с. Совпадающие записи: {', '.join(task['aliases'])}.")
with st.expander("Кадр и изображение для модели"):
    inspect_time = st.slider("Кадр для локального просмотра, с", 0.0, max(0.1, float(duration) - 0.1),
                             min(float(task["original_point_sec"] or 0), max(0.0, duration - 0.1)),
                             step=0.1, key="frame-" + task["task_id"])
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_MSEC, inspect_time * 1000)
    ok, frame = capture.read()
    actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
    capture.release()
    if ok:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_column, model_column = st.columns([3, 1])
        original_column.image(rgb, caption=f"Исходный кадр, {actual:.2f} с", width="stretch")
        model_column.image(letterbox(rgb), caption="Вход кандидата: 112 × 112, увеличен для просмотра", width=224)
        st.caption("Проверьте, различимы ли нужные предметы и действия. Просматривайте без звука; "
                   "не устанавливайте личность человека по внешности.")
if task["original_type"]:
    card = registry()["types"][task["original_type"]]
    st.write("Проверить: " + card["visual_check"])
    st.caption(card["applicability"])
with st.form(task["task_id"], clear_on_submit=False):
    reviewer = st.text_input("Проверяющий")
    state = st.selectbox("Результат проверки", ["unresolved", "confirmed_event", "confirmed_normal_interval"])
    event_type = st.selectbox("Тип события", [None] + allowed,
                              index=1 + allowed.index(task["original_type"]) if task["original_type"] else 0,
                              format_func=lambda x: x or "Не выбран / нормальный интервал")
    mode = st.radio("Временная разметка", ["Точка", "Интервал", "Не установлена"], horizontal=True)
    left, right = st.columns(2)
    start = left.number_input("Начало интервала, с", 0.0, float(duration), 0.0, step=0.1)
    end = right.number_input("Конец интервала, с", 0.0, float(duration), float(duration), step=0.1)
    point = st.number_input("Точка, с", 0.0, float(duration), min(float(task["original_point_sec"] or 0), duration), step=0.1)
    observable = st.text_area("Видимые действия и состояния объектов")
    uncertain = st.text_area("Не видно / спорно / требуется регламент")
    normal_kind = st.multiselect("Контекст для сложных нормальных примеров", [
        "открытие и закрытие сейфа", "работа рядом с деньгами", "корректная передача денег",
        "проход через дверь", "передача смены / доступ", "другое",
    ])
    locally_viewed = st.checkbox("Я просмотрел видео локально и проверил указанный момент или весь интервал")
    if st.form_submit_button("Сохранить отдельную проверку"):
        try:
            revision = append_review(destination, task, {
                "reviewer": reviewer, "review_status": state, "type": event_type,
                "start_sec": start if mode == "Интервал" else None,
                "end_sec": end if mode == "Интервал" else None,
                "point_sec": point if mode == "Точка" else None,
                "observable_actions": observable, "uncertainties": uncertain,
                "normal_context": normal_kind, "locally_viewed": locally_viewed,
            }, allowed, duration)
            st.success("Сохранено. Запись ожидает отдельной проверки перед использованием в обучении.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
with st.expander("Таблица заданий на дополнительную разметку"):
    st.dataframe([{k: t[k] for k in ("video", "original_type", "original_point_sec", "start_sec", "end_sec", "review_status")}
                  for t in tasks], hide_index=True, width="stretch")
