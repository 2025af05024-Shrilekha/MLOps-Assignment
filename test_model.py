import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heartml import config, data
from heartml.features import build_pipeline, build_preprocessor
from heartml.predict import predict_records
from heartml.train import CANDIDATES
from heartml.train import test_metrics as score_model


def _xy(raw_df):
    df = data.clean(raw_df)
    return df[config.FEATURES], df[config.TARGET]


def test_preprocessor_output_has_no_nans_and_expected_width(raw_df):
    X, _ = _xy(raw_df)
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    n_onehot = sum(X[c].nunique() for c in config.CATEGORICAL_FEATURES)
    assert Xt.shape[1] == len(config.NUMERIC_FEATURES) + len(config.BINARY_FEATURES) + n_onehot


def test_preprocessor_scales_numeric(raw_df):
    X, _ = _xy(raw_df)
    Xt = build_preprocessor().fit_transform(X)
    num = Xt[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)


def test_unknown_category_does_not_crash(raw_df, sample_patient):
    X, y = _xy(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    weird = dict(sample_patient, restecg=9)  # unseen code
    assert len(predict_records(model, [weird])) == 1


def test_two_model_families_are_configured():
    families = {type(est) for est, _ in CANDIDATES.values()}
    assert families == {LogisticRegression, RandomForestClassifier}
    for _, grid in CANDIDATES.values():
        assert all(k.startswith("clf__") for k in grid)


def test_models_learn_better_than_chance(raw_df):
    X, y = _xy(raw_df)
    for est, _ in CANDIDATES.values():
        model = build_pipeline(est).fit(X, y)
        assert score_model(model, X, y)["test_roc_auc"] > 0.75


def test_prediction_contract(raw_df, sample_patient):
    X, y = _xy(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    out = predict_records(model, [sample_patient, dict(sample_patient, ca=None, thal=None)])
    for r in out:
        assert set(r) == {"prediction", "label", "risk_probability", "confidence"}
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0
        assert 0.0 <= r["risk_probability"] <= 1.0


def test_joblib_roundtrip(tmp_path, raw_df, sample_patient):
    X, y = _xy(raw_df)
    model = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    p = tmp_path / "m.joblib"
    joblib.dump(model, p)
    again = joblib.load(p)
    assert predict_records(model, [sample_patient]) == predict_records(again, [sample_patient])
