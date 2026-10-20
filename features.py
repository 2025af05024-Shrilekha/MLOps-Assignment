"""Feature engineering: one ColumnTransformer that is shipped inside the model.

Because preprocessing is part of the saved Pipeline, the API sends raw
patient values and gets identical transforms to training - no train/serve skew.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
