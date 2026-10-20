"""Inference helpers shared by the CLI and the API.

CLI:  python -m heartml.predict '{"age": 58, "sex": 1, ...}'
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heartml import config


@lru_cache(maxsize=4)
def load_model(path: str | None = None):
    return joblib.load(Path(path or config.MODEL_FILE))


def predict_records(model, records: list[dict], threshold: float = 0.5) -> list[dict]:
    X = pd.DataFrame(records)[config.FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({
            "prediction": pred,
            "label": "heart_disease" if pred else "no_heart_disease",
            "risk_probability": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return out


if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    recs = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(), recs), indent=2))
