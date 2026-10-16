"""Проверки расчёта и API. Запуск: python -m pytest -q"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.report import overall_result, validate_overall_rules
from engine.scoring import load_config, load_json, run_assessment
from tests.test_scoring import DEMO_IDS
from web.app import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config", "head_data_ai")


def test_overall_rules_valid(config):
    assert validate_overall_rules(config.rules) == []


def test_api_matches_engine(config):
    """API возвращает те же уровни, что и исходный движок (main.py)."""
    data = load_json(ROOT / "synthetic" / "candidates.json")
    engine = {r["candidate_id"]: r for r in run_assessment(config, data)["results"]}
    for cid, expected in engine.items():
        r = client.post(f"/api/candidates/{cid}/evaluate")
        assert r.status_code == 200
        got = {c["id"]: (c["level"], c["gap"], c["category"]["id"]) for c in r.json()["competencies"]}
        assert got == {c["id"]: (c["level"], c["gap"], c["category"]["id"]) for c in expected["competencies"]}


def test_candidate_reports():
    strong = client.post("/api/candidates/CAND-0001/evaluate").json()
    assert len(strong["competencies"]) == 6
    assert strong["overall"]["id"] == "full"
    assert [c["id"] for c in strong["strengths"]] == ["change_management"]
    assert strong["development_areas"] == []

    uneven = client.post("/api/candidates/CAND-0002/evaluate").json()
    assert uneven["overall"]["id"] == "partial"
    assert uneven["development_areas"][0]["id"] == "relationships_influence"
    assert uneven["development_areas"][0]["category"]["id"] == "critical"
    assert uneven["recommendations"][0]["priority"] == "high"
    assert any("TM2" in f["indicator"] for f in uneven["flags"])


def test_overall_capped_by_critical(config):
    comps = [{"level": 4.0, "target": 4.0, "category": {"id": "meets"}}] * 9 + \
            [{"level": 2.0, "target": 4.0, "category": {"id": "critical"}}]
    o = overall_result(comps, config.rules)
    assert o["id"] == "partial" and o["capped_by_critical"] is True


def test_list_and_details():
    r = client.get("/api/candidates")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(DEMO_IDS) and body["position"]["name_en"] == "Head of Data AI"
    assert [c["id"] for c in body["candidates"]] == DEMO_IDS
    d = client.get("/api/candidates/CAND-0002").json()
    assert len(d["competencies"]) == 6
    assert sum(len(c["indicators"]) for c in d["competencies"]) == 21


def test_unknown_candidate():
    for r in (client.get("/api/candidates/NOPE"), client.post("/api/candidates/NOPE/evaluate")):
        assert r.status_code == 404
        assert "не найден" in r.json()["detail"]["message"]
    assert client.get("/candidates/NOPE").status_code == 404


def test_what_if_and_input_errors():
    r = client.post("/api/candidates/CAND-0002/evaluate", json={"scores": {"TM2": {"sjt": 4}}})
    assert r.status_code == 200 and r.json()["what_if"] is True
    assert not any(f["indicator"] == "TM2" for f in r.json()["flags"])

    bad = client.post("/api/candidates/CAND-0002/evaluate", json={"scores": {"PM1": {"sjt": 9}}})
    assert bad.status_code == 422 and bad.json()["detail"]["problems"]
    unknown = client.post("/api/candidates/CAND-0002/evaluate", json={"scores": {"XX9": {"sjt": 3}}})
    assert unknown.status_code == 422
    wrong_method = client.post("/api/candidates/CAND-0002/evaluate", json={"scores": {"TM1": {"sim2": 3}}})
    assert wrong_method.status_code == 422
    assert client.post("/api/candidates/CAND-0002/evaluate", json={"scores": {"PM1": {"sjt": "abc"}}}).status_code == 422
    assert client.post("/api/candidates/CAND-0002/evaluate", json={"foo": 1}).status_code == 422


def test_pages():
    home = client.get("/")
    assert home.status_code == 200
    for text in ("Talent Assessment Pilot", "Демонстрационная система оценки компетенций кандидатов",
                 "Head of Data AI", "Открыть оценку", *DEMO_IDS):
        assert text in home.text
    page = client.get("/candidates/CAND-0001")
    assert page.status_code == 200
    assert "Рассчитать оценку" in page.text and "Вернуться к кандидатам" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
