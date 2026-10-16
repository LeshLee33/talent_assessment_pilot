"""Минимальный набор тестов движка подсчёта (engine/scoring.py).

Запуск:  python -m pytest
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.scoring import (
    AssessmentDataError,
    categorize_gap,
    evaluate_competency,
    load_config,
    load_json,
    load_yaml,
    run_assessment,
    score_indicator,
    validate_candidates,
    validate_competency_model,
    validate_position,
    validate_scoring_rules,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CANDIDATES = ROOT / "synthetic" / "candidates.json"
DEMO_IDS = [f"CAND-{i:04d}" for i in range(1, 9)]

EXPECTED_MODEL = {
    "systems_thinking": ["ST1", "ST2", "ST3"],
    "task_management": ["TM1", "TM2", "TM3", "TM4"],
    "people_management": ["PM1", "PM2", "PM3"],
    "relationships_influence": ["RI1", "RI2", "RI3", "RI4"],
    "managerial_accountability": ["MA1", "MA2", "MA3", "MA4"],
    "change_management": ["CM1", "CM2", "CM3"],
}


# ---------------------------------------------------------------------------
# Фикстуры и помощники
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return load_config(CONFIG_DIR, "head_data_ai")


@pytest.fixture
def rules(config):
    return copy.deepcopy(config.rules)


def full_scores(config, value=3) -> dict:
    """Корректные оценки по всем индикаторам: каждому методу из матрицы ставим value."""
    return {
        ind["id"]: {m: value for m in ind["methods"]}
        for c in config.model["competencies"] for ind in c["indicators"]
    }


def candidates_with(scores: dict, cid="CAND-T001") -> dict:
    return {"candidates": [{"id": cid, "position_id": "head_data_ai", "scores": scores}]}


def indicator(level, assessed=True):
    return {"level": level, "assessed": assessed}


# ---------------------------------------------------------------------------
# 1. Модель компетенций: 6 компетенций и 21 индикатор
# ---------------------------------------------------------------------------

class TestCompetencyModel:
    def test_six_competencies_with_expected_ids(self, config):
        ids = [c["id"] for c in config.model["competencies"]]
        assert ids == list(EXPECTED_MODEL)

    def test_twenty_one_indicators_in_right_competencies(self, config):
        actual = {c["id"]: [i["id"] for i in c["indicators"]] for c in config.model["competencies"]}
        assert actual == EXPECTED_MODEL
        assert sum(len(v) for v in actual.values()) == 21

    def test_every_indicator_has_name_and_two_methods_incl_behavioral(self, config):
        behavioral = {m for m, v in config.rules["methods"].items() if v["group"] == "behavioral"}
        for c in config.model["competencies"]:
            assert c["name"]
            for ind in c["indicators"]:
                assert ind["name"], ind["id"]
                assert len(ind["methods"]) >= 2, ind["id"]
                assert set(ind["methods"]) & behavioral, ind["id"]

    def test_position_has_target_for_every_competency_and_is_preliminary(self, config):
        assert set(config.position["target_levels"]) == set(EXPECTED_MODEL)
        assert config.position["status"] == "preliminary"
        for v in config.position["target_levels"].values():
            assert 1 <= v <= 5

    def test_synthetic_candidates_cover_all_indicators(self, config):
        data = load_json(CANDIDATES)
        assert [c["id"] for c in data["candidates"]] == DEMO_IDS
        assert validate_candidates(data, config.model, config.rules) == []


# ---------------------------------------------------------------------------
# 2. Итоговый балл по правилам проекта
# ---------------------------------------------------------------------------

class TestScoreCalculation:
    def test_indicator_averages_behavioral_methods_then_weights(self, rules):
        # поведенческие: (4 + 5) / 2 = 4.5; итог: 0.7 * 4.5 + 0.3 * 4 = 4.35
        r = score_indicator({"sjt": 4, "sim1": 5, "questionnaire": 4}, rules)
        assert r["behavioral"] == 4.5
        assert r["self_report"] == 4
        assert r["level"] == pytest.approx(4.35)
        assert r["assessed"] is True

    def test_competency_is_mean_of_indicators(self, rules):
        inds = {"A": indicator(4.35), "B": indicator(4.0), "C": indicator(4.35)}
        r = evaluate_competency(inds, 4.0, rules)
        assert r["level"] == pytest.approx(4.23)   # 12.7 / 3 = 4.2333 → 4.23
        assert r["gap"] == pytest.approx(0.23)
        assert r["category"]["id"] == "meets"

    def test_end_to_end_matches_hand_calculation(self, config):
        # CAND-0001, «Системное мышление»:
        #   ST1: sjt 4, sim1 5, q 4 → 0.7*4.5 + 0.3*4 = 4.35
        #   ST2: sjt 4, sim1 4, q 4 → 4.00
        #   ST3: sjt 5, sim2 4, q 4 → 4.35
        #   компетенция = 4.2333 → 4.23; цель 4.0; разрыв +0.23
        report = run_assessment(config, load_json(CANDIDATES))
        cand = next(r for r in report["results"] if r["candidate_id"] == "CAND-0001")
        st = next(c for c in cand["competencies"] if c["id"] == "systems_thinking")
        assert [st["indicators"][i]["level"] for i in ("ST1", "ST2", "ST3")] == [4.35, 4.0, 4.35]
        assert (st["level"], st["target"], st["gap"]) == (4.23, 4.0, 0.23)
        assert st["category"]["name"] == "Соответствует требованиям"

    def test_demo_candidates_regression(self, config):
        report = run_assessment(config, load_json(CANDIDATES))
        got = {
            r["candidate_id"]: {c["id"]: (c["level"], c["category"]["id"]) for c in r["competencies"]}
            for r in report["results"]
        }
        assert got["CAND-0001"]["change_management"] == (4.68, "strength")
        assert got["CAND-0002"]["relationships_influence"] == (2.3, "critical")
        assert got["CAND-0002"]["people_management"] == (2.47, "development")

    def test_report_is_json_serializable_and_has_meta(self, config):
        report = run_assessment(config, load_json(CANDIDATES))
        json.dumps(report, ensure_ascii=False)
        assert report["meta"]["position"]["status"] == "preliminary"
        assert len(report["results"]) == len(DEMO_IDS)


# ---------------------------------------------------------------------------
# 3. Веса 70 / 30
# ---------------------------------------------------------------------------

class TestWeights:
    def test_config_weights_are_70_30(self, rules):
        assert rules["group_weights"] == {"behavioral": 0.7, "self_report": 0.3}

    def test_methods_mapped_to_groups(self, rules):
        groups = {m: v["group"] for m, v in rules["methods"].items()}
        assert groups == {"sjt": "behavioral", "sim1": "behavioral",
                          "sim2": "behavioral", "questionnaire": "self_report"}

    @pytest.mark.parametrize("b, q, expected", [
        (5, 1, 3.8),   # 0.7*5 + 0.3*1
        (1, 5, 2.2),   # 0.7*1 + 0.3*5
        (4, 2, 3.4),
        (3, 3, 3.0),
    ])
    def test_weighted_level(self, rules, b, q, expected):
        assert score_indicator({"sjt": b, "questionnaire": q}, rules)["level"] == pytest.approx(expected)

    def test_weights_are_read_from_config_not_hardcoded(self, rules):
        rules["group_weights"] = {"behavioral": 0.5, "self_report": 0.5}
        assert score_indicator({"sjt": 5, "questionnaire": 1}, rules)["level"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 4. Пропущенная оценка не подменяется нулём
# ---------------------------------------------------------------------------

class TestMissingScores:
    def test_missing_behavioral_method_is_skipped_not_zero(self, rules):
        r = score_indicator({"sim1": 4, "sim2": None, "questionnaire": 4}, rules)
        assert r["behavioral"] == 4          # не (4 + 0) / 2 = 2
        assert r["level"] == pytest.approx(4.0)

    def test_no_behavioral_data_means_not_assessed(self, rules):
        r = score_indicator({"sjt": None, "questionnaire": 4}, rules)
        assert r["level"] is None
        assert r["assessed"] is False
        assert r["flags"] == ["insufficient_data"]

    def test_missing_questionnaire_uses_behavioral_only(self, rules):
        r = score_indicator({"sjt": 4, "questionnaire": None}, rules)
        assert r["level"] == pytest.approx(4.0)   # не 0.7*4 + 0.3*0 = 2.8
        assert "no_self_report" in r["flags"]

    def test_unassessed_indicator_excluded_from_competency_mean(self, rules):
        inds = {"A": indicator(4.0), "B": indicator(4.0), "C": indicator(None, assessed=False)}
        r = evaluate_competency(inds, 4.0, rules)
        assert r["level"] == pytest.approx(4.0)   # не (4 + 4 + 0) / 3 = 2.67
        assert (r["assessed_indicators"], r["total_indicators"]) == (2, 3)

    def test_too_few_assessed_indicators_gives_none_not_zero(self, rules):
        inds = {"A": indicator(4.0), "B": indicator(None, False), "C": indicator(None, False)}
        r = evaluate_competency(inds, 4.0, rules)
        assert r["level"] is None and r["gap"] is None
        assert r["category"]["id"] == "not_assessed"

    def test_demo_candidate_with_null_score(self, config):
        report = run_assessment(config, load_json(CANDIDATES))
        cand = next(r for r in report["results"] if r["candidate_id"] == "CAND-0002")
        tm = next(c for c in cand["competencies"] if c["id"] == "task_management")
        assert tm["indicators"]["TM2"]["level"] is None
        assert tm["assessed_indicators"] == 3

    def test_omitted_method_key_is_an_error_not_silently_zero(self, config):
        scores = full_scores(config)
        del scores["ST1"]["sim1"]
        problems = validate_candidates(candidates_with(scores), config.model, config.rules)
        assert any("ST1" in p and "sim1" in p and "null" in p for p in problems)


# ---------------------------------------------------------------------------
# 5. Флаг расхождения самооценки (порог 1.5)
# ---------------------------------------------------------------------------

class TestDiscrepancyFlag:
    def test_threshold_from_config(self, rules):
        assert rules["self_assessment_discrepancy_threshold"] == 1.5

    @pytest.mark.parametrize("scores, flag", [
        ({"sjt": 2, "sim1": 3, "questionnaire": 4}, "self_assessment_overestimated"),   # +1.5 ровно
        ({"sjt": 1, "questionnaire": 5}, "self_assessment_overestimated"),             # +4
        ({"sjt": 4, "sim1": 5, "questionnaire": 3}, "self_assessment_underestimated"),  # −1.5 ровно
        ({"sjt": 5, "questionnaire": 1}, "self_assessment_underestimated"),            # −4
    ])
    def test_flag_set_at_and_above_threshold(self, rules, scores, flag):
        assert flag in score_indicator(scores, rules)["flags"]

    @pytest.mark.parametrize("scores", [
        {"sjt": 3.51, "questionnaire": 5},     # +1.49
        {"sjt": 5, "questionnaire": 3.51},     # −1.49
        {"sjt": 4, "questionnaire": 5},        # +1
        {"sjt": 4, "questionnaire": 4},        # 0
    ])
    def test_no_flag_below_threshold(self, rules, scores):
        assert score_indicator(scores, rules)["flags"] == []

    def test_no_flag_without_questionnaire(self, rules):
        flags = score_indicator({"sjt": 1, "questionnaire": None}, rules)["flags"]
        assert not any(f.startswith("self_assessment") for f in flags)

    def test_flag_listed_in_candidate_report(self, config):
        report = run_assessment(config, load_json(CANDIDATES))
        cand = next(r for r in report["results"] if r["candidate_id"] == "CAND-0002")
        flagged = {f["indicator"]: f["flags"] for f in cand["flags"]}
        assert flagged["PM3"] == ["self_assessment_overestimated"]   # поведение 1.5, опросник 5


# ---------------------------------------------------------------------------
# 6. Категории на граничных значениях
# ---------------------------------------------------------------------------

class TestGapCategories:
    @pytest.mark.parametrize("gap, expected", [
        (1.5, "strength"),
        (0.5, "strength"),        # ≥ +0.5 — граница включительно
        (0.49, "meets"),
        (0.0, "meets"),
        (-0.49, "meets"),
        (-0.5, "development"),    # −0.5 уже зона развития
        (-1.0, "development"),
        (-1.49, "development"),
        (-1.5, "critical"),       # ≤ −1.5 — критичная
        (-4.0, "critical"),
    ])
    def test_boundaries(self, rules, gap, expected):
        assert categorize_gap(gap, rules)["id"] == expected

    def test_category_names(self, rules):
        names = [categorize_gap(g, rules)["name"] for g in (1, 0, -1, -2)]
        assert names == ["Сильная сторона", "Соответствует требованиям",
                         "Зона развития", "Критичная зона развития"]

    @pytest.mark.parametrize("scores, target, expected_gap, expected", [
        # 0.7*3 + 0.3*4 = 3.3000000000000003 → 3.3; разрыв −0.5 без ошибки округления
        ({"sjt": 3, "questionnaire": 4}, 3.8, -0.5, "development"),
        ({"sjt": 4, "sim1": 5, "questionnaire": 4.5}, 4.0, 0.5, "strength"),
        ({"sjt": 2, "questionnaire": 2}, 3.5, -1.5, "critical"),
    ])
    def test_boundaries_through_full_calculation(self, rules, scores, target, expected_gap, expected):
        ind = score_indicator(scores, rules)
        r = evaluate_competency({"X": ind}, target, rules)
        assert r["gap"] == expected_gap
        assert r["category"]["id"] == expected


# ---------------------------------------------------------------------------
# 7. Некорректные данные → понятные сообщения
# ---------------------------------------------------------------------------

class TestValidationErrors:
    @pytest.mark.parametrize("mutate, message", [
        (lambda s: s["ST1"].update(sjt=7), "ST1.sjt: балл должен быть от 1 до 5 или null, получено 7"),
        (lambda s: s["ST1"].update(sjt=0), "получено 0"),
        (lambda s: s["ST1"].update(sjt="4"), "получено '4'"),
        (lambda s: s["ST1"].update(sjt=True), "получено True"),
        (lambda s: s.pop("TM1"), "нет оценок по индикатору TM1"),
        (lambda s: s.update(XX9={"sjt": 3}), "индикатора XX9 нет в модели компетенций"),
        (lambda s: s["CM1"].update(sim1=3), "методы ['sim1'] не измеряют этот индикатор"),
        (lambda s: s.update(ST1=[4, 4]), "ST1: ожидался объект"),
    ])
    def test_bad_candidate_scores(self, config, mutate, message):
        scores = full_scores(config)
        mutate(scores)
        with pytest.raises(AssessmentDataError) as e:
            run_assessment(config, candidates_with(scores), source="candidates.json")
        assert message in str(e.value)
        assert "candidates.json" in str(e.value)

    def test_all_problems_reported_at_once(self, config):
        scores = full_scores(config)
        scores["ST1"]["sjt"] = 9
        del scores["TM1"]
        with pytest.raises(AssessmentDataError) as e:
            run_assessment(config, candidates_with(scores))
        assert len(e.value.problems) == 2

    def test_duplicate_candidate_id(self, config):
        data = candidates_with(full_scores(config))
        data["candidates"].append(copy.deepcopy(data["candidates"][0]))
        assert "кандидат CAND-T001: id повторяется" in validate_candidates(data, config.model, config.rules)

    def test_empty_candidates(self, config):
        assert validate_candidates({"candidates": []}, config.model, config.rules) == [
            "candidates: список кандидатов отсутствует или пуст"]

    def test_candidate_for_other_position(self, config):
        data = candidates_with(full_scores(config))
        data["candidates"][0]["position_id"] = "head_it"
        with pytest.raises(AssessmentDataError, match="оценивается на позицию «head_it»"):
            run_assessment(config, data)

    @pytest.mark.parametrize("mutate, message", [
        (lambda r: r["group_weights"].update(self_report=0.4), "сумма весов должна быть 1.0"),
        (lambda r: r["group_weights"].pop("behavioral"), "нет веса для группы «behavioral»"),
        (lambda r: r["group_weights"].update(behavioral=1.7), "вес должен быть числом от 0 до 1"),
        (lambda r: r["methods"]["sjt"].update(group="other"), "methods.sjt: поле group"),
        (lambda r: r.update(self_assessment_discrepancy_threshold=-1), "нужно положительное число"),
        (lambda r: r["scale"].update(min=5), "min (5) должен быть меньше max (5)"),
        (lambda r: r["gap_categories"][1].update(min_gap=0.7), "пороги должны строго убывать"),
        (lambda r: r["gap_categories"][-1].update(min_gap=-3), "у последней категории min_gap должен быть null"),
    ])
    def test_bad_scoring_rules(self, rules, mutate, message):
        mutate(rules)
        assert any(message in p for p in validate_scoring_rules(rules))

    @pytest.mark.parametrize("mutate, message", [
        (lambda p: p["target_levels"].pop("people_management"),
         "нет целевого уровня для компетенции «people_management»"),
        (lambda p: p["target_levels"].update(change_management=6), "уровень должен быть числом от 1 до 5, получено 6"),
        (lambda p: p["target_levels"].update(unknown=3), "компетенции «unknown» нет в модели"),
    ])
    def test_bad_position(self, config, mutate, message):
        position = copy.deepcopy(config.position)
        mutate(position)
        assert any(message in p for p in validate_position(position, config.model, config.rules))

    @pytest.mark.parametrize("mutate, message", [
        (lambda m: m["competencies"][0]["indicators"][0]["methods"].update(sim9="primary"),
         "неизвестные методы ['sim9']"),
        (lambda m: m["competencies"][0]["indicators"][0].update(methods={"questionnaire": "secondary"}),
         "нет ни одного поведенческого метода"),
        (lambda m: m["competencies"][1]["indicators"][0].update(id="ST1"), "индикатор «ST1»: id повторяется"),
        (lambda m: m["competencies"][0]["indicators"][0]["methods"].update(sjt="main"),
         "роль метода должна быть primary/secondary"),
    ])
    def test_bad_competency_model(self, config, mutate, message):
        model = copy.deepcopy(config.model)
        mutate(model)
        assert any(message in p for p in validate_competency_model(model, config.rules))

    def test_missing_file(self, tmp_path):
        with pytest.raises(AssessmentDataError, match="файл не найден"):
            load_yaml(tmp_path / "nope.yaml")

    def test_broken_json(self, tmp_path):
        f = tmp_path / "broken.json"
        f.write_text("{bad", encoding="utf-8")
        with pytest.raises(AssessmentDataError, match=r"некорректный JSON: строка 1, столбец 2"):
            load_json(f)

    def test_broken_yaml(self, tmp_path):
        f = tmp_path / "broken.yaml"
        f.write_text("a: [1, 2", encoding="utf-8")
        with pytest.raises(AssessmentDataError, match="некорректный YAML"):
            load_yaml(f)

    def test_unknown_position(self):
        with pytest.raises(AssessmentDataError, match=r"head_it\.yaml.*\n.*файл не найден"):
            load_config(CONFIG_DIR, "head_it")

    def test_cli_prints_error_and_exits_with_code_1(self, tmp_path):
        bad = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        bad["candidates"][0]["scores"]["ST1"]["sjt"] = 7
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--candidates", str(f), "--output", str(tmp_path / "out.json")],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert proc.returncode == 1
        assert "ОШИБКА ДАННЫХ" in proc.stderr
        assert "ST1.sjt: балл должен быть от 1 до 5 или null, получено 7" in proc.stderr
        assert not (tmp_path / "out.json").exists()

    def test_cli_success_writes_json(self, tmp_path):
        out = tmp_path / "out.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--output", str(out)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stderr
        assert len(json.loads(out.read_text(encoding="utf-8"))["results"]) == len(DEMO_IDS)
