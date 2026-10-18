"""Forte-inspired presentation. Assets are local; no external requests or tracking."""

import base64
import html
import json
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
TYPE_LABELS = {
    "cash_area_door_left_open": "Открытая дверь кассовой зоны",
    "cash_left_unattended": "Деньги без присмотра",
    "cash_taken_before_verification": "Деньги приняты до проверки",
    "customer_cash_mixing": "Смешивание денег клиентов",
    "incorrect_cash_handover": "Некорректная передача денег",
    "overlapping_customer_transactions": "Пересечение операций клиентов",
    "personal_items_in_work_area": "Личные вещи в рабочей зоне",
    "safe_key_left_in_lock": "Ключ оставлен в замке сейфа",
    "safe_left_open": "Открытый сейф",
    "unauthorized_person_in_cash_area": "Доступ в кассовую зону требует проверки",
    "unclassified_violation": "Неклассифицированное нарушение",
}


def style():
    fonts = []
    for row in json.loads((ASSETS / "fonts.json").read_text()):
        encoded = base64.b64encode((ASSETS / row["file"]).read_bytes()).decode()
        fonts.append(
            "@font-face{font-family:'Forte Open Sans';font-style:normal;"
            "font-weight:300 800;font-display:swap;src:url(data:font/woff2;base64,"
            + encoded
            + ") format('woff2');unicode-range:"
            + row["unicode_range"]
            + ";}"
        )
    return "<style>" + "".join(fonts) + (ASSETS / "forte.css").read_text() + "</style>"


def markup(value):
    st.markdown(value, unsafe_allow_html=True)


def header(ready):
    logo = base64.b64encode((ASSETS / "forte-logo.svg").read_bytes()).decode()
    status = "Готова к анализу" if ready else "Сервис недоступен"
    markup(f"""<header class="forte-topbar">
      <a class="forte-brand" href="#" aria-label="ForteAI — главная">
        <img src="data:image/svg+xml;base64,{logo}" alt="Forte" width="108" height="32">
        <span class="forte-brand-divider"></span><span class="forte-product">Cash Control</span>
      </a>
      <div class="forte-top-right"><span class="forte-project">Проект ForteAI</span>
      <span class="forte-connection {"is-ready" if ready else "is-offline"}">
        <span class="forte-dot"></span>{status}</span></div>
    </header>""")


def hero():
    markup("""<section class="forte-hero">
      <div class="forte-hero-copy"><div class="forte-eyebrow">ВИДЕОАНАЛИТИКА КАССОВЫХ ОПЕРАЦИЙ</div>
        <h1>Внимание к деталям.<br>Контроль в ваших руках.</h1>
        <p>Проверяйте записи кассовых операций.<br>Подозрительные фрагменты, таймкоды и кадры — в одном отчёте.</p>
      </div>
      <div class="forte-hero-art" aria-hidden="true">
        <svg viewBox="0 0 400 235" xmlns="http://www.w3.org/2000/svg" fill="none">
          <circle cx="304" cy="90" r="96" fill="#DDD5DD"/>
          <path d="M340 0a90 90 0 0 1 60 85" stroke="#83DCEC" stroke-width="33"/>
          <rect x="44" y="34" width="290" height="183" rx="22" fill="#D4C8D0" transform="rotate(-8 44 34)"/>
          <rect x="57" y="18" width="286" height="183" rx="22" fill="white"/>
          <rect x="72" y="33" width="256" height="124" rx="13" fill="#F5E7ED"/>
          <path d="M85 62V48h15M300 48h15v14M85 126v15h15M300 141h15v-15" stroke="#A31551" stroke-width="2" stroke-linecap="round"/>
          <circle cx="200" cy="94" r="30" fill="#4A0221"/>
          <path d="m194 82 19 12-19 12V82Z" fill="white"/>
          <path d="M82 178h234" stroke="#E5E0E4" stroke-width="4" stroke-linecap="round"/>
          <path d="M82 178h84" stroke="#A31551" stroke-width="4" stroke-linecap="round"/>
          <circle cx="166" cy="178" r="6" fill="#A31551"/>
          <rect x="239" y="178" width="132" height="42" rx="13" fill="#4A0221"/>
          <path d="m255 200 5 5 10-11" stroke="#83DCEC" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M282 195h64M282 204h43" stroke="white" stroke-opacity=".75" stroke-width="3" stroke-linecap="round"/>
          <circle cx="43" cy="171" r="20" fill="#A31551"/>
          <path d="M43 162v18M34 171h18" stroke="#FFF" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
    </section>""")


def section_label(number, text):
    markup(
        f'<div class="forte-eyebrow forte-section-label"><span>{html.escape(number)}</span> {html.escape(text)}</div>'
    )


def report_guide():
    markup("""<div class="forte-guide"><div class="forte-eyebrow">ПОСЛЕ АНАЛИЗА</div>
      <h2>Всё, что нужно<br>для проверки</h2>
      <div class="forte-guide-row"><span>01</span><div><h3>Фрагменты для внимания</h3><p>Возможные нарушения с указанием типа события.</p></div></div>
      <div class="forte-guide-row"><span>02</span><div><h3>Таймкоды и кадры</h3><p>Момент срабатывания и реальные кадры из записи.</p></div></div>
      <div class="forte-guide-row"><span>03</span><div><h3>Контекст для решения</h3><p>Пояснения и основания, которые проверяет оператор.</p></div></div>
    </div>""")


def empty_result():
    markup("""<div class="forte-empty"><div class="forte-empty-icon" aria-hidden="true">
      <svg width="28" height="30" viewBox="0 0 28 30" fill="none"><path d="M6 2h11l6 6v20H6V2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M17 2v7h6M10 15h9M10 20h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></div>
      <h3>Здесь появится результат</h3><p>Добавьте видео и запустите анализ.<br>Когда обработка завершится, здесь появится отчёт.</p></div>""")


def footer():
    markup("""<footer class="forte-footer"><span><b>ForteAI</b> · Cash Control</span>
      <span>Экспериментальный проект · Решение остаётся за оператором</span></footer>""")
