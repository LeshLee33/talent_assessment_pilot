"""Итоговый отчёт по кандидату: общий результат, сильные стороны, зоны развития, рекомендации.

Работает поверх результата engine.scoring.assess_candidate и не пересчитывает баллы.
Пороги общего результата читаются из раздела overall_result в config/scoring_rules.yaml.
Тексты рекомендаций — детерминированные шаблоны (без LLM): они зависят только от
категорий компетенций, уровней индикаторов и флагов.
"""

from __future__ import annotations

from statistics import mean

from engine.scoring import Config, _is_number

DEVELOPMENT_CATEGORIES = ("development", "critical")
STRENGTH_CATEGORY = "strength"

LIMITATIONS = [
    "Демонстрация методики на синтетических данных: кандидаты вымышленные.",
    "Результат не подтверждает реальные качества человека и не используется для кадровых решений.",
    "Целевые уровни позиции предварительные и не согласованы с HR.",
    "Правило общего результата и тексты рекомендаций — допущения прототипа (шаблоны, без LLM).",
    "AI рекомендует — решение принимает человек.",
]


def validate_overall_rules(rules: dict) -> list[str]:
    """Проверка раздела overall_result в scoring_rules.yaml."""
    o = rules.get("overall_result")
    if not isinstance(o, dict):
        return ["overall_result: раздел отсутствует"]
    problems: list[str] = []
    cat_ids = {c["id"] for c in rules["gap_categories"]}
    match = o.get("match_categories")
    if not isinstance(match, list) or not match or not set(match) <= cat_ids:
        problems.append(f"overall_result.match_categories: нужен непустой список из {sorted(cat_ids)}")
    levels = o.get("levels")
    if not isinstance(levels, list) or not levels:
        return problems + ["overall_result.levels: нужен непустой список"]
    prev = None
    for i, lv in enumerate(levels):
        share = lv.get("min_match_share") if isinstance(lv, dict) else None
        if not isinstance(lv, dict) or not lv.get("id") or not lv.get("name") or not _is_number(share):
            problems.append(f"overall_result.levels[{i}]: нужны поля id, name и числовой min_match_share")
            continue
        if prev is not None and share >= prev:
            problems.append(f"overall_result.levels[{i}]: пороги должны строго убывать")
        prev = share
    if not problems and levels[-1]["min_match_share"] > 0:
        problems.append("overall_result.levels: у последнего уровня min_match_share должен быть 0")
    if o.get("critical_category") is not None and o["critical_category"] not in cat_ids:
        problems.append("overall_result.critical_category: неизвестная категория")
    if o.get("cap_if_critical") is not None and o["cap_if_critical"] not in {lv.get("id") for lv in levels if isinstance(lv, dict)}:
        problems.append("overall_result.cap_if_critical: неизвестный уровень")
    return problems


def overall_result(competencies: list[dict], rules: dict) -> dict:
    """Общий результат: средние уровни, доля компетенций на уровне цели и итоговая формулировка."""
    o = rules["overall_result"]
    nd = rules["rounding"]
    assessed = [c for c in competencies if c["level"] is not None]
    if not assessed:
        return {"id": "not_assessed", "name": "Недостаточно данных", "average_level": None,
                "average_target": None, "average_gap": None, "match_share": None,
                "matched": 0, "assessed": 0, "total": len(competencies), "capped_by_critical": False}

    matched = sum(1 for c in assessed if c["category"]["id"] in o["match_categories"])
    share = matched / len(assessed)
    levels = o["levels"]
    idx = next(i for i, lv in enumerate(levels) if share >= lv["min_match_share"])

    capped = False
    has_critical = any(c["category"]["id"] == o.get("critical_category") for c in assessed)
    if has_critical and o.get("cap_if_critical"):
        cap_idx = next(i for i, lv in enumerate(levels) if lv["id"] == o["cap_if_critical"])
        if idx < cap_idx:
            idx, capped = cap_idx, True

    avg_level = round(mean(c["level"] for c in assessed), nd)
    avg_target = round(mean(c["target"] for c in assessed), nd)
    return {"id": levels[idx]["id"], "name": levels[idx]["name"],
            "average_level": avg_level, "average_target": avg_target,
            "average_gap": round(avg_level - avg_target, nd),
            "match_share": round(share, nd), "matched": matched,
            "assessed": len(assessed), "total": len(competencies), "capped_by_critical": capped}


def _weakest_indicators(comp: dict, limit: int = 2) -> list[str]:
    inds = [(iid, i) for iid, i in comp["indicators"].items() if i["assessed"]]
    inds.sort(key=lambda x: (x[1]["level"], x[0]))
    return [f"{iid} «{i['name']}» ({i['level']:.2f})" for iid, i in inds[:limit]]


def build_recommendations(competencies: list[dict], flags: list[dict]) -> list[dict]:
    """Детерминированные рекомендации по шаблонам. Порядок: критичные зоны, зоны развития,
    флаги данных, сильные стороны."""
    recs: list[dict] = []
    for c in sorted((c for c in competencies if c["category"]["id"] == "critical"), key=lambda c: c["gap"]):
        recs.append({"priority": "high", "competency": c["name"], "text": (
            f"Приоритетно: индивидуальный план развития по компетенции «{c['name']}» "
            f"(разрыв {c['gap']:+.2f}). Начать с индикаторов: {', '.join(_weakest_indicators(c))}. "
            "Рекомендуется повторная оценка поведенческими методами после программы развития.")})
    for c in sorted((c for c in competencies if c["category"]["id"] == "development"), key=lambda c: c["gap"]):
        recs.append({"priority": "medium", "competency": c["name"], "text": (
            f"Развивающие задачи и обратная связь по компетенции «{c['name']}» "
            f"(разрыв {c['gap']:+.2f}). Фокус: {', '.join(_weakest_indicators(c))}.")})

    over = [f["indicator"] for f in flags if "self_assessment_overestimated" in f["flags"]]
    under = [f["indicator"] for f in flags if "self_assessment_underestimated" in f["flags"]]
    missing = [f["indicator"] for f in flags if "insufficient_data" in f["flags"]]
    if over:
        recs.append({"priority": "medium", "competency": None, "text": (
            f"Самооценка выше поведенческих результатов по индикаторам {', '.join(over)}: "
            "провести развивающую беседу с разбором результатов SJT и симуляций.")})
    if under:
        recs.append({"priority": "low", "competency": None, "text": (
            f"Самооценка ниже поведенческих результатов по индикаторам {', '.join(under)}: "
            "обсудить с кандидатом подтверждённые сильные проявления.")})
    if missing:
        recs.append({"priority": "medium", "competency": None, "text": (
            f"Недостаточно поведенческих данных по индикаторам {', '.join(missing)}: "
            "дособрать данные перед интерпретацией результата.")})
    for c in competencies:
        if c["category"]["id"] == STRENGTH_CATEGORY:
            recs.append({"priority": "low", "competency": c["name"], "text": (
                f"Использовать компетенцию «{c['name']}» как ресурс: наставничество, "
                "участие в проектах, где она критична.")})
    if not recs:
        recs.append({"priority": "low", "competency": None, "text": (
            "Существенных разрывов с целевым профилем нет. Поддерживать уровень, "
            "периодически подтверждать оценку.")})
    return recs


def build_candidate_report(assessment: dict, candidate: dict, config: Config, meta: dict) -> dict:
    """Собирает отчёт для веб-интерфейса и API из результата assess_candidate."""
    comps = assessment["competencies"]
    scale = config.rules["scale"]

    def brief(c: dict) -> dict:
        return {"id": c["id"], "name": c["name"], "level": c["level"],
                "target": c["target"], "gap": c["gap"], "category": c["category"]}

    return {
        "candidate_id": assessment["candidate_id"],
        "profile_note": assessment["profile_note"],
        "position": {**meta["position"], "name_en": config.position.get("name_en")},
        "scale": {"min": scale["min"], "max": scale["max"],
                  "labels": {str(k): v for k, v in (scale.get("labels") or {}).items()}},
        "methods": {mid: m.get("name", mid) for mid, m in config.rules["methods"].items()},
        "competencies": comps,
        "overall": overall_result(comps, config.rules),
        "strengths": [brief(c) for c in comps if c["category"]["id"] == STRENGTH_CATEGORY],
        "meets": [brief(c) for c in comps if c["category"]["id"] == "meets"],
        "development_areas": sorted((brief(c) for c in comps if c["category"]["id"] in DEVELOPMENT_CATEGORIES),
                                    key=lambda c: c["gap"]),
        "not_assessed": [brief(c) for c in comps if c["level"] is None],
        "flags": assessment["flags"],
        "recommendations": build_recommendations(comps, assessment["flags"]),
        "meta": {k: meta[k] for k in ("generated_at", "competency_model_version",
                                      "scoring_rules_version", "group_weights", "disclaimer")},
        "limitations": LIMITATIONS,
        "input_scores": candidate["scores"],
    }
