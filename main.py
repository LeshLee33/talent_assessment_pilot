"""Демо: расчёт профиля компетенций вымышленных кандидатов относительно позиции.

Запуск:  python main.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.scoring import AssessmentDataError, load_config, load_json, run_assessment

ROOT = Path(__file__).resolve().parent

FLAG_LABELS = {
    "insufficient_data": "недостаточно данных — индикатор не оценён",
    "no_self_report": "нет самооценки — учтены только поведенческие методы",
    "self_assessment_overestimated": "завышенная самооценка",
    "self_assessment_underestimated": "заниженная самооценка",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Демо AI-оценки кандидатов в кадровый резерв (синтетические данные)")
    p.add_argument("--position", default="head_data_ai", help="id позиции (файл config/positions/<id>.yaml)")
    p.add_argument("--candidates", type=Path, default=ROOT / "synthetic" / "candidates.json")
    p.add_argument("--config-dir", type=Path, default=ROOT / "config")
    p.add_argument("--output", type=Path, default=None, help="куда сохранить JSON (по умолчанию output/results_<position>.json)")
    return p.parse_args()


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def print_report(report: dict) -> None:
    meta = report["meta"]
    pos = meta["position"]
    w = meta["group_weights"]
    print("=" * 84)
    print(f"Позиция: {pos['name']}  [{pos['status']}]")
    if pos["status_note"]:
        print(f"  ! {pos['status_note']}")
    print(f"Веса: поведенческие методы {w['behavioral']:.0%}, опросник {w['self_report']:.0%}")
    print(meta["disclaimer"])

    for r in report["results"]:
        print("=" * 84)
        print(f"Кандидат {r['candidate_id']}  ({r['profile_note']})")
        print("-" * 84)
        print(f"{'Компетенция':<34}{'Уровень':>8}{'Цель':>7}{'Разрыв':>8}   Категория")
        for c in r["competencies"]:
            gap = "—" if c["gap"] is None else f"{c['gap']:+.2f}"
            print(f"{c['name']:<34}{fmt(c['level']):>8}{c['target']:>7.1f}{gap:>8}   {c['category']['name']}")
        if r["flags"]:
            print("Флаги:")
            for f in r["flags"]:
                labels = ", ".join(FLAG_LABELS.get(x, x) for x in f["flags"])
                print(f"  - {f['indicator']} {f['name']}: {labels}")
    print("=" * 84)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config_dir, args.position)
        candidates = load_json(args.candidates)
        report = run_assessment(config, candidates, source=str(args.candidates))
    except AssessmentDataError as e:
        print(f"ОШИБКА ДАННЫХ. {e}", file=sys.stderr)
        return 1

    output = args.output or ROOT / "output" / f"results_{args.position}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(report)
    print(f"Результат сохранён: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
