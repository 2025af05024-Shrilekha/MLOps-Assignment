"""Central configuration: paths, column definitions and constants.

Keeping every column list in one place means the training code, the API
schema and the tests can never drift apart.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_META_FILE = MODELS_DIR / "model_meta.json"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Same 303 Cleveland records, re-published with text labels (ISLR "Heart.csv").
# Only used if the UCI server is unreachable; converted back to UCI coding.
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

# The 14 columns of processed.cleveland.data, in file order.
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous / count features -> impute median + standard scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Already 0/1 -> impute most frequent, pass through
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> impute most frequent + one-hot
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-risk-classifier")
