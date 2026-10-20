import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heartml import config, data
from heartml.features import build_pipeline


@pytest.fixture
def client(tmp_path, raw_df, monkeypatch):
    df = data.clean(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(df[config.FEATURES], df[config.TARGET])
    mp = tmp_path / "model.joblib"
    joblib.dump(model, mp)
    meta = {"model_name": "test_lr", "sklearn_version": "x"}
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    import api.main as m
    monkeypatch.setattr(m, "MODEL_PATH", str(mp))
    monkeypatch.setattr(m, "META_PATH", str(tmp_path / "model_meta.json"))
    with TestClient(m.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test_lr"
    assert "x-request-id" in r.headers


def test_predict_allows_missing_ca_thal(client, sample_patient):
    p = dict(sample_patient)
    p.pop("ca")
    p.pop("thal")
    assert client.post("/predict", json=p).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("cp", 9), ("sex", 2), ("chol", 5000)])
def test_predict_rejects_invalid_input(client, sample_patient, field, value):
    assert client.post("/predict", json=dict(sample_patient, **{field: value})).status_code == 422


def test_batch(client, sample_patient):
    r = client.post("/predict/batch", json=[sample_patient] * 3)
    assert r.status_code == 200 and len(r.json()) == 3


def test_metrics_endpoint_counts_predictions(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_request_latency_seconds_bucket" in text
    summary = client.get("/metrics/summary").json()
    assert sum(summary["predictions"].values()) >= 1
