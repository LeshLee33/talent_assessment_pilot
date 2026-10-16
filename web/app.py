"""Веб-интерфейс и API демо Talent Assessment Pilot (FastAPI + Jinja2).

Запуск:  uvicorn web.app:app --host 127.0.0.1 --port 8000

Расчёт полностью переиспользует engine.scoring (баллы индикаторов и компетенций)
и engine.report (общий результат, сильные стороны, зоны развития, рекомендации).
Данные читаются из config/ и synthetic/candidates.json при каждом запросе,
поэтому правки конфигов видны без перезапуска.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from engine.report import LIMITATIONS, build_candidate_report, validate_overall_rules
from engine.scoring import (
    DISCLAIMER,
    AssessmentDataError,
    Config,
    assess_candidate,
    load_config,
    load_json,
    validate_candidates,
)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CANDIDATES_PATH = ROOT / "synthetic" / "candidates.json"
POSITION_ID = "head_data_ai"

app = FastAPI(
    title="Talent Assessment Pilot",
    description="Демонстрационная система оценки компетенций кандидатов (только синтетические данные).",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


class EvaluateRequest(BaseModel):
    """Необязательные изменения баллов для расчёта «что если».

    Пример: {"scores": {"PM1": {"sjt": 3}}} — меняет балл SJT по индикатору PM1,
    остальные баллы берутся из synthetic/candidates.json. Файл с данными не изменяется.
    """

    model_config = ConfigDict(extra="forbid")
    scores: dict[str, dict[str, float | None]] | None = None


class DataError(Exception):
    def __init__(self, status: int, message: str, problems: list[str] | None = None):
        self.status, self.message, self.problems = status, message, problems or []


@app.exception_handler(DataError)
async def _data_error_handler(_: Request, exc: DataError) -> JSONResponse:
    return JSONResponse(status_code=exc.status,
                        content={"detail": {"message": exc.message, "problems": exc.problems}})


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_context() -> tuple[Config, dict]:
    """Конфигурация методики и синтетические кандидаты. Ошибки данных → HTTP 500."""
    try:
        config = load_config(CONFIG_DIR, POSITION_ID)
        problems = validate_overall_rules(config.rules)
        if problems:
            raise AssessmentDataError("config/scoring_rules.yaml", problems)
        data = load_json(CANDIDATES_PATH)
        problems = validate_candidates(data, config.model, config.rules)
        if problems:
            raise AssessmentDataError("synthetic/candidates.json", problems)
    except AssessmentDataError as e:
        raise DataError(500, f"Ошибка конфигурации или тестовых данных в «{e.source}»", e.problems) from e
    return config, data


def find_candidate(data: dict, candidate_id: str) -> dict:
    for c in data["candidates"]:
        if c["id"] == candidate_id:
            return c
    raise DataError(404, f"Кандидат с ID «{candidate_id}» не найден")


def position_info(config: Config) -> dict:
    p = config.position
    return {"id": p["id"], "name": p["name"], "name_en": p.get("name_en"),
            "status": p["status"], "status_note": p.get("status_note", "")}


def candidate_summary(c: dict) -> dict:
    return {"id": c["id"], "profile_note": c.get("profile_note", ""), "position_id": c.get("position_id")}


def candidate_details(c: dict, config: Config) -> dict:
    """Исходные данные кандидата, сгруппированные по модели компетенций."""
    targets = config.position["target_levels"]
    return {
        **candidate_summary(c),
        "position": position_info(config),
        "scale": {"min": config.rules["scale"]["min"], "max": config.rules["scale"]["max"]},
        "methods": {mid: m.get("name", mid) for mid, m in config.rules["methods"].items()},
        "competencies": [
            {"id": comp["id"], "name": comp["name"], "target": targets[comp["id"]],
             "indicators": [{"id": ind["id"], "name": ind["name"], "scores": c["scores"][ind["id"]]}
                            for ind in comp["indicators"]]}
            for comp in config.model["competencies"]
        ],
        "disclaimer": DISCLAIMER,
    }


def merge_scores(stored: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(stored)
    for iid, methods in overrides.items():
        merged[iid] = {**merged[iid], **methods} if iid in merged else methods
    return merged


def evaluate(candidate: dict, config: Config, overrides: dict | None = None) -> dict:
    cand = dict(candidate)
    if overrides:
        cand["scores"] = merge_scores(candidate["scores"], overrides)
        problems = validate_candidates({"candidates": [cand]}, config.model, config.rules)
        if problems:
            raise DataError(422, "Некорректные входные баллы", problems)
    assessment = assess_candidate(cand, config)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "position": position_info(config),
        "competency_model_version": config.model.get("version"),
        "scoring_rules_version": config.rules.get("version"),
        "group_weights": config.rules["group_weights"],
        "disclaimer": DISCLAIMER,
    }
    report = build_candidate_report(assessment, cand, config, meta)
    report["what_if"] = bool(overrides)
    return report


# ---------------------------------------------------------------------------
# HTML-страницы
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Главная страница")
def index(request: Request):
    try:
        config, data = load_context()
    except DataError as e:
        return templates.TemplateResponse(request, "error.html",
                                          {"message": e.message, "problems": e.problems}, status_code=e.status)
    return templates.TemplateResponse(request, "index.html", {
        "position": position_info(config),
        "candidates": [candidate_summary(c) for c in data["candidates"]],
        "competencies": [{"name": c["name"], "target": config.position["target_levels"][c["id"]]}
                         for c in config.model["competencies"]],
        "disclaimer": DISCLAIMER,
    })


@app.get("/candidates/{candidate_id}", response_class=HTMLResponse, summary="Страница оценки кандидата")
def candidate_page(request: Request, candidate_id: str):
    try:
        config, data = load_context()
        find_candidate(data, candidate_id)
    except DataError as e:
        return templates.TemplateResponse(request, "error.html",
                                          {"message": e.message, "problems": e.problems}, status_code=e.status)
    return templates.TemplateResponse(request, "candidate.html", {
        "candidate_id": candidate_id,
        "position": position_info(config),
        "limitations": LIMITATIONS,
    })


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

_ERRORS = {404: {"description": "Кандидат не найден"},
           500: {"description": "Ошибка конфигурации или тестовых данных"}}


@app.get("/api/candidates", summary="Список кандидатов", responses={500: _ERRORS[500]})
def api_candidates():
    config, data = load_context()
    return {"position": position_info(config), "count": len(data["candidates"]),
            "candidates": [candidate_summary(c) for c in data["candidates"]],
            "disclaimer": DISCLAIMER}


@app.get("/api/candidates/{candidate_id}", summary="Данные кандидата", responses=_ERRORS)
def api_candidate(candidate_id: str):
    config, data = load_context()
    return candidate_details(find_candidate(data, candidate_id), config)


@app.post("/api/candidates/{candidate_id}/evaluate", summary="Расчёт оценки и отчёт",
          responses={**_ERRORS, 422: {"description": "Некорректные входные баллы"}})
def api_evaluate(candidate_id: str, payload: EvaluateRequest | None = Body(default=None)):
    config, data = load_context()
    candidate = find_candidate(data, candidate_id)
    return jsonable_encoder(evaluate(candidate, config, payload.scores if payload else None))


@app.get("/api/health", summary="Проверка работоспособности")
def health():
    load_context()
    return {"status": "ok"}


@app.exception_handler(404)
async def _not_found(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/") or request.url.path.startswith("/static/"):
        return JSONResponse(status_code=404, content={"detail": {"message": "Ресурс не найден", "problems": []}})
    return templates.TemplateResponse(request, "error.html",
                                      {"message": "Страница не найдена", "problems": []}, status_code=404)
