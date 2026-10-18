"""Offline normative references. A classifier hypothesis is not an observed fact."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def registry():
    return json.loads(Path(__file__).with_name("rule_registry.json").read_text())


def validate_registry(allowed_types):
    data = registry()
    if set(data["types"]) != set(allowed_types):
        raise ValueError("Rule registry and model vocabulary differ")
    for card in data["types"].values():
        for ref in card["references"]:
            if ref["source"] not in data["sources"]:
                raise ValueError("Unknown normative source")
    return data


def describe_hypothesis(event_type, start, end, score, threshold):
    """Describe the check, never invent persons, actions, objects or observations."""
    data = registry()
    card = data["types"][event_type]
    refs = []
    for ref in card["references"]:
        source = data["sources"][ref["source"]]
        refs.append(f"{source['short_title']}, {ref['point']} ({source['url']}#{ref['anchor']})")
    basis = "; ".join(refs) if refs else "Прямое нормативное основание не установлено"
    return (
        f"Гипотеза модели: {card['title']}. Фрагмент {start:.1f}–{end:.1f} с. "
        f"Оценка модели {score:.3f}, порог {threshold:.3f}; оценка не является вероятностью нарушения. "
        f"Проверить по видео: {card['visual_check']} "
        f"Нормативная справка: {basis}. {card['applicability']} "
        "Действия и применимость нормы автоматически не подтверждены. "
        "Кадры взяты из загруженного видео; таймкод находится внутри окна и не обозначает границу события."
    )
