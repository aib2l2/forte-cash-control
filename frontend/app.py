"""Streamlit is an API client; inference belongs exclusively to the backend."""

import base64
import time

import httpx
import streamlit as st

from backend.config import Settings

st.set_page_config(page_title="Forte Cash Control", page_icon="🔎", layout="wide")
st.title("Проверка кассовых операций")
st.caption("ForteAI · Datathon · локальная обработка видео")
st.warning(
    "Это каркас проекта. AI-модель ещё не подключена: загрузка и API работают, "
    "но сервис пока не определяет нарушения."
)

try:
    settings = Settings.from_env()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

headers = {"X-API-Key": settings.api_key}
st.session_state.setdefault("job_active", False)

with st.sidebar:
    st.subheader("Подключение")
    st.code(settings.api_base_url, language=None)
    try:
        response = httpx.get(f"{settings.api_base_url}/health", timeout=2, trust_env=False)
        response.raise_for_status()
        st.success("Сервер доступен")
    except httpx.HTTPError:
        st.error("Сервер недоступен. Запустите Backend в VS Code.")
    st.caption("Проверка доступности сервера не означает готовность AI-модели.")

video = st.file_uploader("Выберите видео .mov", type=["mov"])
st.caption(f"Лимит API: {settings.max_upload_mb} МБ. Звук не используется.")

if st.button(
    "Запустить анализ",
    type="primary",
    disabled=video is None or st.session_state.job_active,
):
    if video.size > settings.max_upload_mb * 1024 * 1024:
        st.error("Файл превышает лимит API.")
    else:
        try:
            with st.spinner("Загружаем видео…"):
                video.seek(0)
                response = httpx.post(
                    f"{settings.api_base_url}/api/v1/analyze",
                    headers=headers,
                    files={"file": (video.name, video, "video/quicktime")},
                    timeout=60,
                    trust_env=False,
                )
                response.raise_for_status()
            st.session_state.job_id = response.json()["job_id"]
            st.session_state.submitted_name = video.name
            st.session_state.submitted_at = time.monotonic()
            st.session_state.job_active = True
            st.session_state.pop("job_result", None)
            st.rerun()
        except httpx.HTTPStatusError as exc:
            st.error(exc.response.json().get("detail", "Сервер отклонил загрузку."))
        except (httpx.HTTPError, ValueError, KeyError):
            st.error("Не удалось загрузить видео. Проверьте сервер и повторите попытку.")


@st.fragment(run_every=1.0 if st.session_state.job_active else None)
def show_job() -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return
    st.subheader(f"Результат: {st.session_state.submitted_name}")
    st.caption(f"Задача: {job_id}")
    result = st.session_state.get("job_result")
    if result is None:
        try:
            response = httpx.get(
                f"{settings.api_base_url}/api/v1/jobs/{job_id}",
                headers=headers,
                timeout=5,
                trust_env=False,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError):
            st.error("Не удалось получить статус. Повторяем проверку автоматически.")
            return

    if result["status"] == "processing":
        elapsed = int(time.monotonic() - st.session_state.submitted_at)
        st.info(f"Обработка… Прошло {elapsed} с.")
        if elapsed >= 300:
            st.warning("Превышен лимит жюри в 5 минут. Проверка статуса продолжается.")
        return

    if st.session_state.job_active:
        st.session_state.job_result = result
        st.session_state.job_active = False
        st.rerun()

    if result["status"] == "failed":
        st.error(result.get("error", "Обработка завершилась ошибкой."))
        st.caption("Ошибка обработки не означает, что нарушений нет.")
    elif result["verdict"] == "no_violation":
        st.success("Нарушение не обнаружено")
    else:
        st.warning("Обнаружены признаки возможных нарушений")
        for index, event in enumerate(result["violations"], start=1):
            with st.container(border=True):
                st.subheader(f"{index}. {event['type']}")
                st.write(f"Таймкод: {event['timecode_sec']:.2f} с")
                st.write(event["description"])
                for frame in event["evidence_frames"]:
                    st.image(
                        base64.b64decode(frame["image_base64"]),
                        caption=f"Кадр: {frame['timestamp_sec']:.2f} с",
                    )
    with st.expander("Ответ API"):
        st.json(result)


show_job()
