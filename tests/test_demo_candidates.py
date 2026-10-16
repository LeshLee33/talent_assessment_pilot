"""Сценарии синтетических кандидатов: каждый проверяет свой случай методики.

Ожидаемые значения — по правилам config/scoring_rules.yaml (веса 70/30, пороги категорий,
overall_result по доле компетенций на уровне цели с ограничением при критичной зоне).
"""

import pytest
from fastapi.testclient import TestClient

from tests.test_scoring import DEMO_IDS
from web.app import app

client = TestClient(app)


def evaluate(cid: str) -> dict:
    r = client.post(f"/api/candidates/{cid}/evaluate")
    assert r.status_code == 200, r.text
    return r.json()


def categories(report: dict) -> dict:
    return {c["id"]: (c["gap"], c["category"]["id"]) for c in report["competencies"]}


@pytest.mark.parametrize("cid, overall, capped", [
    ("CAND-0001", "full", False),       # сильный
    ("CAND-0002", "partial", False),    # неровный: доля ровно 0.5 — граница уровня partial
    ("CAND-0003", "partial", False),    # средний
    ("CAND-0004", "full", False),       # заниженная самооценка
    ("CAND-0005", "full", False),       # неполные данные: доля по 5 оценённым из 6
    ("CAND-0006", "partial", False),    # граничный: доля ровно 0.5
    ("CAND-0007", "low", False),        # слабый
    ("CAND-0008", "partial", True),     # одна критичная зона ограничивает «full»
])
def test_overall_result(cid, overall, capped):
    o = evaluate(cid)["overall"]
    assert (o["id"], o["capped_by_critical"]) == (overall, capped)


def test_all_overall_levels_covered():
    got = {evaluate(cid)["overall"]["id"] for cid in DEMO_IDS}
    assert got == {"full", "partial", "low"}


def test_average_candidate():
    r = evaluate("CAND-0003")
    assert [c["id"] for c in r["development_areas"]] == ["change_management", "relationships_influence"]
    assert all(c["category"]["id"] == "development" for c in r["development_areas"])
    assert r["overall"]["match_share"] == 0.67


def test_underestimated_self_assessment():
    r = evaluate("CAND-0004")
    flagged = {f["indicator"] for f in r["flags"]}
    assert flagged == {"TM1", "TM2", "TM3", "TM4", "PM1", "PM2", "PM3"}
    assert all(f["flags"] == ["self_assessment_underestimated"] for f in r["flags"])
    assert {c["id"] for c in r["strengths"]} == {"task_management", "people_management", "managerial_accountability"}
    assert any("ниже поведенческих результатов" in rec["text"] for rec in r["recommendations"])


def test_incomplete_data():
    r = evaluate("CAND-0005")
    cm = next(c for c in r["competencies"] if c["id"] == "change_management")
    assert cm["level"] is None and cm["category"]["id"] == "not_assessed"
    assert (cm["assessed_indicators"], cm["total_indicators"]) == (1, 3)     # 1/3 < 50%
    assert [c["id"] for c in r["not_assessed"]] == ["change_management"]
    assert (r["overall"]["assessed"], r["overall"]["total"]) == (5, 6)
    assert r["overall"]["average_target"] == 3.8                             # цель тоже по 5 оценённым
    ri2 = next(c for c in r["competencies"] if c["id"] == "relationships_influence")["indicators"]["RI2"]
    assert ri2["flags"] == ["no_self_report"] and ri2["level"] == ri2["behavioral"]


def test_boundary_candidate():
    """Границы категорий на реальных данных: +0.5, −0.5, −1.5 и +0.4."""
    assert categories(evaluate("CAND-0006")) == {
        "systems_thinking": (0.5, "strength"),
        "task_management": (-0.5, "development"),
        "people_management": (-1.5, "critical"),
        "relationships_influence": (0.0, "meets"),
        "managerial_accountability": (-0.5, "development"),
        "change_management": (0.4, "meets"),
    }


def test_weak_candidate():
    r = evaluate("CAND-0007")
    assert r["overall"]["match_share"] == 0.17
    assert len(r["development_areas"]) == 5
    assert not any(c["category"]["id"] == "critical" for c in r["competencies"])


def test_single_critical_zone_caps_result():
    r = evaluate("CAND-0008")
    assert r["overall"]["match_share"] == 0.83            # без ограничения было бы «full»
    assert [(c["id"], c["category"]["id"]) for c in r["development_areas"]] == [
        ("relationships_influence", "critical")]
    assert r["recommendations"][0]["priority"] == "high"


@pytest.mark.parametrize("cid", DEMO_IDS)
def test_page_opens(cid):
    assert client.get(f"/candidates/{cid}").status_code == 200
