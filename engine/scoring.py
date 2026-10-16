"""Детерминированный подсчёт уровней компетенций и сравнение с профилем позиции.

Все веса, пороги и шкала читаются из config/scoring_rules.yaml.
В этом модуле нет «магических чисел» методики: только логика агрегации
(план пилота, раздел 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

# Группы методов — часть схемы конфига, а не настраиваемые значения:
# поведенческие методы обязательны для оценки индикатора,
# самооценка сравнивается с ними для флага расхождения.
BEHAVIORAL = "behavioral"
SELF_REPORT = "self_report"
REQUIRED_GROUPS = (BEHAVIORAL, SELF_REPORT)

DISCLAIMER = (
    "Демо на синтетических данных. Баллы не используются для кадровых решений. "
    "AI рекомендует — решение принимает человек."
)


class AssessmentDataError(Exception):
    """Ошибка во входных данных или конфигурации. Содержит список всех найденных проблем."""

    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        details = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Ошибки в «{source}» ({len(problems)}):\n{details}")


@dataclass
class Config:
    model: dict
    rules: dict
    position: dict


# ---------------------------------------------------------------------------
# Загрузка файлов
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise AssessmentDataError(str(path), ["файл не найден"])
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AssessmentDataError(str(path), [f"некорректный YAML: {e}"]) from e
    if not isinstance(data, dict):
        raise AssessmentDataError(str(path), ["ожидался YAML-словарь на верхнем уровне"])
    return data


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise AssessmentDataError(str(path), ["файл не найден"])
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise AssessmentDataError(
            str(path), [f"некорректный JSON: строка {e.lineno}, столбец {e.colno}: {e.msg}"]
        ) from e
    if not isinstance(data, dict):
        raise AssessmentDataError(str(path), ["ожидался JSON-объект на верхнем уровне"])
    return data


def load_config(config_dir: Path, position_id: str) -> Config:
    rules = load_yaml(config_dir / "scoring_rules.yaml")
    _raise_if(validate_scoring_rules(rules), "config/scoring_rules.yaml")

    model = load_yaml(config_dir / "competency_model.yaml")
    _raise_if(validate_competency_model(model, rules), "config/competency_model.yaml")

    position_path = config_dir / "positions" / f"{position_id}.yaml"
    position = load_yaml(position_path)
    _raise_if(validate_position(position, model, rules), f"config/positions/{position_id}.yaml")

    return Config(model=model, rules=rules, position=position)


def _raise_if(problems: list[str], source: str) -> None:
    if problems:
        raise AssessmentDataError(source, problems)


# ---------------------------------------------------------------------------
# Проверка корректности
# ---------------------------------------------------------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_scoring_rules(rules: dict) -> list[str]:
    problems: list[str] = []

    scale = rules.get("scale")
    if not isinstance(scale, dict) or not _is_number(scale.get("min")) or not _is_number(scale.get("max")):
        problems.append("scale: нужны числовые поля min и max")
    elif scale["min"] >= scale["max"]:
        problems.append(f"scale: min ({scale['min']}) должен быть меньше max ({scale['max']})")

    weights = rules.get("group_weights")
    if not isinstance(weights, dict):
        problems.append("group_weights: раздел отсутствует")
    else:
        for g in REQUIRED_GROUPS:
            if g not in weights:
                problems.append(f"group_weights: нет веса для группы «{g}»")
            elif not _is_number(weights[g]) or not 0 <= weights[g] <= 1:
                problems.append(f"group_weights.{g}: вес должен быть числом от 0 до 1, получено {weights[g]!r}")
        extra = set(weights) - set(REQUIRED_GROUPS)
        if extra:
            problems.append(f"group_weights: неизвестные группы {sorted(extra)}; допустимы {list(REQUIRED_GROUPS)}")
        if not problems and abs(sum(weights.values()) - 1.0) > 1e-9:
            problems.append(f"group_weights: сумма весов должна быть 1.0, получено {sum(weights.values())}")

    methods = rules.get("methods")
    if not isinstance(methods, dict) or not methods:
        problems.append("methods: раздел отсутствует или пуст")
    else:
        for mid, m in methods.items():
            if not isinstance(m, dict) or m.get("group") not in REQUIRED_GROUPS:
                problems.append(f"methods.{mid}: поле group должно быть одним из {list(REQUIRED_GROUPS)}")

    thr = rules.get("self_assessment_discrepancy_threshold")
    if not _is_number(thr) or thr <= 0:
        problems.append("self_assessment_discrepancy_threshold: нужно положительное число")

    share = rules.get("min_assessed_indicator_share")
    if not _is_number(share) or not 0 < share <= 1:
        problems.append("min_assessed_indicator_share: нужно число в интервале (0, 1]")

    if not isinstance(rules.get("rounding"), int) or rules["rounding"] < 0:
        problems.append("rounding: нужно целое неотрицательное число")

    cats = rules.get("gap_categories")
    if not isinstance(cats, list) or len(cats) < 2:
        problems.append("gap_categories: нужен список минимум из двух категорий")
    else:
        prev = None
        for i, c in enumerate(cats):
            label = f"gap_categories[{i}]"
            if not isinstance(c, dict) or not c.get("id") or not c.get("name"):
                problems.append(f"{label}: нужны поля id и name")
                continue
            last = i == len(cats) - 1
            mg = c.get("min_gap")
            if last:
                if mg is not None:
                    problems.append(f"{label} ({c['id']}): у последней категории min_gap должен быть null")
            elif not _is_number(mg):
                problems.append(f"{label} ({c['id']}): min_gap должен быть числом")
            else:
                if prev is not None and mg >= prev:
                    problems.append(
                        f"{label} ({c['id']}): пороги должны строго убывать сверху вниз ({mg} >= {prev})"
                    )
                prev = mg
    return problems


def validate_competency_model(model: dict, rules: dict) -> list[str]:
    problems: list[str] = []
    known_methods = set(rules["methods"])
    behavioral = {m for m, v in rules["methods"].items() if v["group"] == BEHAVIORAL}
    comps = model.get("competencies")
    if not isinstance(comps, list) or not comps:
        return ["competencies: список компетенций отсутствует или пуст"]

    comp_ids: set[str] = set()
    ind_ids: set[str] = set()
    for ci, c in enumerate(comps):
        cid = c.get("id") if isinstance(c, dict) else None
        if not cid:
            problems.append(f"competencies[{ci}]: нет id")
            continue
        if cid in comp_ids:
            problems.append(f"компетенция «{cid}»: id повторяется")
        comp_ids.add(cid)
        if not c.get("name"):
            problems.append(f"компетенция «{cid}»: нет name")
        inds = c.get("indicators")
        if not isinstance(inds, list) or not inds:
            problems.append(f"компетенция «{cid}»: нет индикаторов")
            continue
        for ii, ind in enumerate(inds):
            iid = ind.get("id") if isinstance(ind, dict) else None
            if not iid:
                problems.append(f"компетенция «{cid}», индикатор [{ii}]: нет id")
                continue
            if iid in ind_ids:
                problems.append(f"индикатор «{iid}»: id повторяется")
            ind_ids.add(iid)
            methods = ind.get("methods")
            if not isinstance(methods, dict) or not methods:
                problems.append(f"индикатор «{iid}»: не указаны методы")
                continue
            unknown = set(methods) - known_methods
            if unknown:
                problems.append(f"индикатор «{iid}»: неизвестные методы {sorted(unknown)}")
            bad_roles = {m: r for m, r in methods.items() if r not in ("primary", "secondary")}
            if bad_roles:
                problems.append(f"индикатор «{iid}»: роль метода должна быть primary/secondary, получено {bad_roles}")
            # Методология 3.5: минимум два метода, из них минимум один поведенческий
            if len(methods) < 2:
                problems.append(f"индикатор «{iid}»: нужно минимум 2 метода, указано {len(methods)}")
            if not set(methods) & behavioral:
                problems.append(f"индикатор «{iid}»: нет ни одного поведенческого метода")
    return problems


def validate_position(position: dict, model: dict, rules: dict) -> list[str]:
    problems: list[str] = []
    for field in ("id", "name", "status"):
        if not position.get(field):
            problems.append(f"нет поля {field}")
    targets = position.get("target_levels")
    if not isinstance(targets, dict):
        return problems + ["target_levels: раздел отсутствует"]

    lo, hi = rules["scale"]["min"], rules["scale"]["max"]
    comp_ids = {c["id"] for c in model["competencies"]}
    for cid in sorted(comp_ids - set(targets)):
        problems.append(f"target_levels: нет целевого уровня для компетенции «{cid}»")
    for cid in sorted(set(targets) - comp_ids):
        problems.append(f"target_levels: компетенции «{cid}» нет в модели компетенций")
    for cid, v in targets.items():
        if not _is_number(v) or not lo <= v <= hi:
            problems.append(f"target_levels.{cid}: уровень должен быть числом от {lo} до {hi}, получено {v!r}")
    return problems


def validate_candidates(data: dict, model: dict, rules: dict) -> list[str]:
    problems: list[str] = []
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return ["candidates: список кандидатов отсутствует или пуст"]

    lo, hi = rules["scale"]["min"], rules["scale"]["max"]
    indicator_methods = {
        ind["id"]: set(ind["methods"]) for c in model["competencies"] for ind in c["indicators"]
    }
    seen: set[str] = set()
    for i, cand in enumerate(cands):
        cid = cand.get("id") if isinstance(cand, dict) else None
        if not cid:
            problems.append(f"candidates[{i}]: нет id")
            continue
        if cid in seen:
            problems.append(f"кандидат {cid}: id повторяется")
        seen.add(cid)
        scores = cand.get("scores")
        if not isinstance(scores, dict):
            problems.append(f"кандидат {cid}: нет раздела scores")
            continue
        for iid in sorted(set(indicator_methods) - set(scores)):
            problems.append(f"кандидат {cid}: нет оценок по индикатору {iid}")
        for iid in sorted(set(scores) - set(indicator_methods)):
            problems.append(f"кандидат {cid}: индикатора {iid} нет в модели компетенций")
        for iid, s in scores.items():
            if iid not in indicator_methods:
                continue
            if not isinstance(s, dict):
                problems.append(f"кандидат {cid}, {iid}: ожидался объект вида {{метод: балл}}")
                continue
            expected = indicator_methods[iid]
            missing, extra = expected - set(s), set(s) - expected
            if missing:
                problems.append(
                    f"кандидат {cid}, {iid}: нет оценок методами {sorted(missing)} "
                    f"(если данных нет — укажите null)"
                )
            if extra:
                problems.append(
                    f"кандидат {cid}, {iid}: методы {sorted(extra)} не измеряют этот индикатор по матрице"
                )
            for m, v in s.items():
                if v is not None and (not _is_number(v) or not lo <= v <= hi):
                    problems.append(f"кандидат {cid}, {iid}.{m}: балл должен быть от {lo} до {hi} или null, получено {v!r}")
    return problems


# ---------------------------------------------------------------------------
# Подсчёт
# ---------------------------------------------------------------------------

def score_indicator(method_scores: dict[str, float | None], rules: dict) -> dict:
    """Балл индикатора = взвешенное среднее поведенческих методов и самооценки."""
    nd = rules["rounding"]
    by_group: dict[str, list[float]] = {BEHAVIORAL: [], SELF_REPORT: []}
    for m, v in method_scores.items():
        if v is not None:
            by_group[rules["methods"][m]["group"]].append(v)

    behavioral = mean(by_group[BEHAVIORAL]) if by_group[BEHAVIORAL] else None
    self_report = mean(by_group[SELF_REPORT]) if by_group[SELF_REPORT] else None
    flags: list[str] = []

    if behavioral is None:
        # Без поведенческих данных индикатор не оцениваем (раздел 4.2, п. 4)
        return {"level": None, "behavioral": None,
                "self_report": _round(self_report, nd),
                "assessed": False, "flags": ["insufficient_data"]}

    if self_report is None:
        level = behavioral
        flags.append("no_self_report")
    else:
        w = rules["group_weights"]
        level = w[BEHAVIORAL] * behavioral + w[SELF_REPORT] * self_report
        diff = round(self_report - behavioral, nd)
        thr = rules["self_assessment_discrepancy_threshold"]
        if diff >= thr:
            flags.append("self_assessment_overestimated")
        elif diff <= -thr:
            flags.append("self_assessment_underestimated")

    return {"level": _round(level, nd), "behavioral": _round(behavioral, nd),
            "self_report": _round(self_report, nd), "assessed": True, "flags": flags}


def categorize_gap(gap: float, rules: dict) -> dict:
    """Категория по разрыву (факт − цель) согласно gap_categories."""
    cats = rules["gap_categories"]
    for c in cats[:-1]:
        passed = gap >= c["min_gap"] if c.get("inclusive") else gap > c["min_gap"]
        if passed:
            return {"id": c["id"], "name": c["name"]}
    return {"id": cats[-1]["id"], "name": cats[-1]["name"]}


def evaluate_competency(indicator_results: dict[str, dict], target: float, rules: dict) -> dict:
    """Уровень компетенции (среднее по оценённым индикаторам), разрыв с целью и категория."""
    nd = rules["rounding"]
    levels = [r["level"] for r in indicator_results.values() if r["assessed"]]
    share = len(levels) / len(indicator_results)

    if not levels or share < rules["min_assessed_indicator_share"]:
        return {"level": None, "target": target, "gap": None,
                "category": {"id": "not_assessed", "name": "Недостаточно данных"},
                "assessed_indicators": len(levels), "total_indicators": len(indicator_results)}

    level = round(mean(levels), nd)
    gap = round(level - target, nd)
    return {"level": level, "target": target, "gap": gap,
            "category": categorize_gap(gap, rules),
            "assessed_indicators": len(levels), "total_indicators": len(indicator_results)}


def assess_candidate(candidate: dict, config: Config) -> dict:
    rules, targets = config.rules, config.position["target_levels"]
    competencies = []
    for comp in config.model["competencies"]:
        indicators = {}
        for ind in comp["indicators"]:
            r = score_indicator(candidate["scores"][ind["id"]], rules)
            indicators[ind["id"]] = {"name": ind["name"], **r}
        result = evaluate_competency(indicators, targets[comp["id"]], rules)
        competencies.append({"id": comp["id"], "name": comp["name"], **result, "indicators": indicators})

    summary: dict[str, list[str]] = {}
    for c in competencies:
        summary.setdefault(c["category"]["id"], []).append(c["name"])
    flagged = [
        {"indicator": iid, "name": ind["name"], "flags": ind["flags"]}
        for c in competencies for iid, ind in c["indicators"].items() if ind["flags"]
    ]
    return {"candidate_id": candidate["id"], "profile_note": candidate.get("profile_note", ""),
            "competencies": competencies, "summary_by_category": summary, "flags": flagged}


def run_assessment(config: Config, candidates_data: dict, source: str = "candidates") -> dict:
    pos = config.position
    problems = validate_candidates(candidates_data, config.model, config.rules)
    for cand in candidates_data.get("candidates") or []:
        if isinstance(cand, dict) and cand.get("position_id") not in (None, pos["id"]):
            problems.append(
                f"кандидат {cand.get('id')}: оценивается на позицию «{cand['position_id']}», "
                f"а расчёт запущен для «{pos['id']}»"
            )
    _raise_if(problems, source)
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "position": {"id": pos["id"], "name": pos["name"], "status": pos["status"],
                         "status_note": pos.get("status_note", "")},
            "competency_model_version": config.model.get("version"),
            "scoring_rules_version": config.rules.get("version"),
            "group_weights": config.rules["group_weights"],
            "disclaimer": DISCLAIMER,
        },
        "results": [assess_candidate(c, config) for c in candidates_data["candidates"]],
    }


def _round(x: float | None, nd: int) -> float | None:
    return None if x is None else round(x, nd)
