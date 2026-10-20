"""Heart-risk prediction API (FastAPI).

Endpoints
  GET  /           service info
  GET  /health     liveness/readiness (503 until the model is loaded)
  GET  /model      metadata of the served model
  POST /predict    one patient  -> prediction + confidence
  POST /predict/batch  list of patients
  GET  /metrics    Prometheus metrics (scraped by Prometheus)
  GET  /metrics/summary  human-readable JSON view of the same counters
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from heartml.predict import load_model, predict_records

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "model.joblib"))
META_PATH = os.getenv("MODEL_META_PATH", str(Path(MODEL_PATH).with_name("model_meta.json")))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
POD = os.getenv("HOSTNAME", socket.gethostname())


# ---------- structured JSON request logging ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                "logger": record.name, "msg": record.getMessage(), "pod": POD}
        base.update(getattr(record, "extra_fields", {}))
        return json.dumps(base)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart_api")
log.handlers = [_handler]
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
log.propagate = False


# ---------- Prometheus metrics (own registry -> clean for tests) ----------
REGISTRY = CollectorRegistry()
REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "endpoint", "status"],
                   registry=REGISTRY)
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5), registry=REGISTRY)
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served by class", ["label"],
                      registry=REGISTRY)
RISK = Histogram("heart_api_risk_probability", "Distribution of predicted risk probability",
                 buckets=[i / 10 for i in range(1, 11)], registry=REGISTRY)
ERRORS = Counter("heart_api_prediction_errors_total", "Failed predictions", registry=REGISTRY)
MODEL_INFO = Gauge("heart_api_model_info", "Served model (value always 1)",
                   ["model_name", "sklearn_version"], registry=REGISTRY)
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if model loaded", registry=REGISTRY)

STATE: dict = {"model": None, "meta": {}, "started": time.time()}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        STATE["model"] = load_model(MODEL_PATH)
        if Path(META_PATH).exists():
            STATE["meta"] = json.loads(Path(META_PATH).read_text())
        MODEL_INFO.labels(STATE["meta"].get("model_name", "unknown"),
                          STATE["meta"].get("sklearn_version", "unknown")).set(1)
        MODEL_LOADED.set(1)
        log.info("model loaded", extra={"extra_fields": {"model_path": MODEL_PATH}})
    except Exception as exc:  # keep serving /health=503 so k8s can see it
        MODEL_LOADED.set(0)
        log.error("model load failed", extra={"extra_fields": {"error": str(exc)}})
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        if endpoint != "/metrics":  # don't let scrapes drown the signal
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": rid, "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


# ---------- schemas ----------
class Patient(BaseModel):
    """Raw clinical values, UCI Cleveland coding. ca/thal may be null (imputed)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
        "thalach": 130, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7}})

    age: int = Field(..., ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4] = Field(..., description="chest pain type")
    trestbps: float = Field(..., gt=50, lt=260, description="resting BP mm Hg")
    chol: float = Field(..., gt=80, lt=700, description="serum cholesterol mg/dl")
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(..., gt=50, lt=250, description="max heart rate")
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: int | None = Field(None, ge=0, le=3)
    thal: Literal[3, 6, 7] | None = None


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    risk_probability: float
    confidence: float
    model_name: str
    threshold: float


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return STATE["model"]


def _predict(patients: list[Patient]) -> list[dict]:
    model = _require_model()
    try:
        res = predict_records(model, [p.model_dump() for p in patients], THRESHOLD)
    except Exception as exc:
        ERRORS.inc()
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    name = STATE["meta"].get("model_name", "unknown")
    for r in res:
        PREDICTIONS.labels(r["label"]).inc()
        RISK.observe(r["risk_probability"])
        r.update(model_name=name, threshold=THRESHOLD)
    return res


# ---------- routes ----------
@app.get("/")
def root():
    return {"service": "heart-risk-api", "pod": POD, "docs": "/docs",
            "endpoints": ["/health", "/model", "/predict", "/predict/batch", "/metrics",
                          "/metrics/summary"]}


@app.get("/health")
def health(response: Response):
    ok = STATE["model"] is not None
    response.status_code = 200 if ok else 503
    return {"status": "ok" if ok else "model_not_loaded", "pod": POD,
            "uptime_s": round(time.time() - STATE["started"], 1)}


@app.get("/model")
def model_info():
    _require_model()
    m = STATE["meta"]
    return {k: m.get(k) for k in ("model_name", "best_params", "test_metrics", "cv_metrics",
                                  "trained_at", "sklearn_version", "mlflow_run_id")}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return _predict([patient])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(patients: list[Patient]):
    if not patients or len(patients) > 500:
        raise HTTPException(status_code=422, detail="send 1-500 patients")
    return _predict(patients)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
def metrics_summary():
    """Same numbers as /metrics, as JSON - handy for demos without Grafana."""
    out = {"pod": POD, "uptime_s": round(time.time() - STATE["started"], 1),
           "requests": {}, "predictions": {}, "prediction_errors": 0}
    for fam in REGISTRY.collect():
        for s in fam.samples:
            if s.name == "heart_api_requests_total":
                key = f'{s.labels["method"]} {s.labels["endpoint"]} {s.labels["status"]}'
                out["requests"][key] = int(s.value)
            elif s.name == "heart_api_predictions_total":
                out["predictions"][s.labels["label"]] = int(s.value)
            elif s.name == "heart_api_prediction_errors_total":
                out["prediction_errors"] = int(s.value)
    return out
