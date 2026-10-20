"""Shared fixtures. Tests use a synthetic UCI-shaped dataset so they run
offline and fast; the real data is exercised by the training CI job."""
import numpy as np
import pandas as pd
import pytest

from heartml import config


def _synthetic_raw(n=160, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    # make the label learnable: risk rises with cp=4, exang, oldpeak, low thalach
    score = (df.cp == 4) * 1.5 + df.exang + df.oldpeak / 2 - (df.thalach - 150) / 30
    df["num"] = np.where(score + rng.normal(0, 0.7, n) > 1.2, rng.integers(1, 5, n), 0)
    df.loc[:3, "ca"] = np.nan
    df.loc[4:5, "thal"] = np.nan
    return df[config.UCI_COLUMNS]


@pytest.fixture
def raw_df():
    return _synthetic_raw()


@pytest.fixture
def sample_patient():
    return {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
            "restecg": 2, "thalach": 130, "exang": 1, "oldpeak": 2.4, "slope": 2,
            "ca": 2, "thal": 7}
