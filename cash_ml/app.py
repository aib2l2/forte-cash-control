"""Forte-styled Streamlit client for the existing local video analysis API."""

import base64
import json
import time

import httpx
import streamlit as st

from backend.config import Settings
from cash_ml import forte_ui as ui

st.set_page_config(
    page_title="ForteAI · Cash Control", page_icon=":material/shield:", layout="wide"
)
ui.markup(ui.style())
settings = Settings.from_env()
headers = {"X-API-Key": settings.api_key}

# Keep existing jobs and the production API contract across presentation updates.
if st.session_state.get("model_format") != 2:
    for key in ("active", "job", "result", "started"):
        st.session_state.pop(key, None)
    st.session_state.model_format = 2
st.session_state.setdefault("active", False)
status, ready = {}, False
try:
    response = httpx.get(
        settings.api_base_url + "/api/v1/model", headers=headers, timeout=5, trust_env=False
    )
    response.raise_for_status()
    status = response.json()
    ready = bool(status["ready"])
except (httpx.HTTPError, ValueError, KeyError):
    pass

ui.header(ready)
analysis_tab, model_tab = st.tabs(["Проверка видео", "О модели и качестве"])


@st.fragment(run_every=2.0 if st.session_state.active else None)
def show_result():
    with st.container(key="result_card"):
        ui.section_label("02", "РЕЗУЛЬТАТ ПРОВЕРКИ")
        if "job" not in st.session_state:
            ui.empty_result()
            return
        result = st.session_state.get("result")
        if result is None:
            try:
                response = httpx.get(
                    settings.api_base_url + "/api/v1/jobs/" + st.session_state.job,
                    headers=headers,
                    timeout=5,
                    trust_env=False,
                )
                response.raise_for_status()
                result = response.json()
            except (httpx.HTTPError, ValueError, KeyError):
                st.error("Не удалось получить статус. Проверяем соединение с сервером…")
                return
        if result["status"] == "processing":
            st.subheader("Видео в обработке")
            elapsed = max(
                0, int(time.monotonic() - st.session_state.get("started", time.monotonic()))
            )
            st.info(f"Анализируем видео… {elapsed} с", icon=":material/hourglass_top:")
            st.caption(
                "Можно оставить страницу открытой: отчёт появится автоматически. Лимит с учётом очереди — пять минут."
            )
            return
        if st.session_state.active:
            st.session_state.result = result
            st.session_state.active = False
            st.rerun()
        if result["status"] == "failed":
            st.subheader("Не удалось завершить анализ")
            st.error(result.get("error", "Ошибка обработки видео."))
            st.caption(
                "Попробуйте загрузить запись ещё раз. Ошибка обработки не означает отсутствие нарушений."
            )
        elif result["verdict"] == "no_violation":
            st.subheader("Нарушение не обнаружено")
            st.success("Модель не выделила подозрительных фрагментов.", icon=":material/task_alt:")
            st.caption(
                "Это не гарантирует отсутствие нарушений. При необходимости запись проверяет оператор."
            )
        else:
            st.subheader("Обнаружены признаки возможных нарушений")
            st.caption(
                "Проверьте события по кадрам и контексту записи. Срабатывание модели — повод для проверки."
            )
            for index, event in enumerate(result["violations"], 1):
                with st.container(border=True):
                    seconds = float(event["timecode_sec"])
                    timestamp = f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
                    ui.markup(
                        f'<div class="forte-event-header"><span class="forte-event-number">СОБЫТИЕ {index:02d}</span>'
                        f'<span class="forte-timecode">{timestamp}</span></div>'
                    )
                    st.subheader(ui.TYPE_LABELS.get(event["type"], event["type"]))
                    st.write(event["description"])
                    frames = event["evidence_frames"]
                    columns = st.columns(max(1, min(3, len(frames))), gap="small")
                    for column, frame in zip(columns, frames, strict=False):
                        with column:
                            st.image(
                                base64.b64decode(frame["image_base64"]),
                                caption=f"Кадр {frame['timestamp_sec']:.2f} с",
                                width="stretch",
                            )
                    st.caption(f"Тип в API: {event['type']}")
        if result["status"] == "done":
            st.download_button(
                "Скачать отчёт JSON",
                json.dumps(result, ensure_ascii=False, indent=2),
                file_name="forte-analysis.json",
                mime="application/json",
                icon=":material/download:",
                on_click="ignore",
            )
        with st.expander("Ответ API"):
            st.json(result)


with analysis_tab:
    ui.hero()
    ui.markup(
        f'<div class="forte-facts"><span><i></i>Формат .MOV</span><span><i></i>До {settings.max_upload_mb} МБ</span>'
        "<span><i></i>До 5 минут на обработку</span></div>"
    )
    upload_column, guide_column = st.columns([1.65, 1], gap="large")
    with upload_column, st.container(key="upload_card"):
        ui.section_label("01", "НОВАЯ ПРОВЕРКА")
        st.subheader("Загрузите видео")
        st.caption("Добавьте запись кассовой операции для анализа.")
        video = st.file_uploader(
            "Выберите видео .mov",
            type=["mov"],
            max_upload_size=settings.max_upload_mb,
            label_visibility="collapsed",
            disabled=st.session_state.active,
        )
        if video is not None:
            st.caption(f"Выбран файл: {video.name} · {video.size / (1024 * 1024):.1f} МБ")
        if not ready:
            st.error("Сервис анализа сейчас недоступен. Повторите попытку позже.")
        if st.button(
            "Запустить анализ",
            type="primary",
            width="stretch",
            icon=":material/play_arrow:",
            disabled=not ready or video is None or st.session_state.active,
        ):
            if video.size > settings.max_upload_mb * 1024 * 1024:
                st.error("Превышен лимит размера файла")
            else:
                try:
                    video.seek(0)
                    response = httpx.post(
                        settings.api_base_url + "/api/v1/analyze",
                        headers=headers,
                        files={"file": (video.name, video, "video/quicktime")},
                        timeout=60,
                        trust_env=False,
                    )
                    response.raise_for_status()
                    st.session_state.job = response.json()["job_id"]
                    st.session_state.started = time.monotonic()
                    st.session_state.active = True
                    st.session_state.pop("result", None)
                    st.rerun()
                except (httpx.HTTPError, ValueError, KeyError):
                    st.error("Не удалось отправить видео. Проверьте соединение и размер файла.")
        st.caption("Видео обрабатывается внутри сервиса. Облачные AI API не используются.")
    with guide_column, st.container(key="report_guide"):
        ui.report_guide()
    ui.markup(
        '<div class="forte-notice"><span>ⓘ</span><span><b>Помощник оператора · экспериментальная версия.</b> '
        "Тип события и основания для вывода необходимо проверить по записи.</span></div>"
    )
    show_result()

with model_tab:
    with st.container(key="model_summary"):
        ui.section_label("FORTEAI", "МОДЕЛЬ И ОГРАНИЧЕНИЯ")
        st.subheader("Прозрачность каждого результата")
        st.write(
            "Сервис выделяет фрагменты для проверки. Окончательное решение принимает оператор, учитывая полную запись и правила работы."
        )
        ui.markup(
            f'<div class="forte-model-tags"><span>R3D-18</span><span>Активная версия: v2</span>'
            f"<span>{len(status.get('allowed_types', []))} типов из разметки</span></div>"
        )
        st.caption(
            "Набор данных: 42 записи, 40 уникальных видео. Результаты на обучающих видео не подтверждают качество на новых записях."
        )
    if ready:
        with st.expander("Параметры активной модели"):
            st.write(f"Порог срабатывания: {status['threshold']:.3f}")
            st.caption(
                "Порог не является точностью модели. Оценки не являются калиброванными вероятностями."
            )
            st.write("Подключена исходная модель v2. Новые кандидаты не заменяют её автоматически.")
        with st.expander("Допустимые типы из Excel"):
            for event_type in status["allowed_types"]:
                st.write(ui.TYPE_LABELS.get(event_type, event_type))
                st.caption(event_type)
        experiment = status.get("latest_experiment")
        if experiment and experiment["state"] == "running":
            st.info("Идёт сравнение новых моделей. Анализ использует исходную v2.")
        elif experiment and experiment["state"] == "complete":
            st.info(
                "Для анализа сохранена исходная v2: надёжное улучшение на независимых сценах не подтверждено."
            )
            with st.expander("Сравнение на одинаковых группах сцен"):
                rows = []
                for key, label in [
                    ("baseline", "Исходный подход, переобучен"),
                    ("candidate", "Новый кандидат"),
                ]:
                    result = experiment["comparison"][key]["test"]
                    matrix = result["detection"]["confusion_matrix_tn_fp_fn_tp"]
                    rows.append(
                        {
                            "Модель": label,
                            "Видео": result["detection"]["videos"],
                            "Balanced accuracy": result["detection"]["balanced_accuracy"],
                            "Ложные тревоги (видео)": matrix[0][1],
                            "Пропуски (видео)": matrix[1][0],
                            "Тип в размеченной точке": result["type_accuracy_at_points"],
                            "Верный тип в пределах 4 с": result["timing"]["typed_point_recall"],
                        }
                    )
                st.dataframe(rows, hide_index=True, width="stretch")
                st.caption(
                    "Повторно используемая проверка: 5 видео из одной группы похожих сцен. Это исследовательское сравнение, а не оценка качества на новых камерах. В Excel размечены точки; точность начала и конца события проверить нельзя."
                )
                st.json(
                    {"support": experiment["supports"], "deployment": experiment["deployment_gate"]}
                )
        with st.expander("Исторические результаты исходной v2"):
            st.json(status["test"])
            st.json(status["type_test"])
            st.caption(
                "Старое разделение не учитывало похожие серии. Этот тест уже использовался; он не является независимой финальной проверкой."
            )

ui.footer()
