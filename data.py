"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset.

Run:  python -m heartml.data
"""
from __future__ import annotations

import io
import logging
import urllib.request

import numpy as np
import pandas as pd

from heartml import config

log = logging.getLogger("heartml.data")

# Text label -> UCI numeric code, used only for the mirror fallback.
_CP_MAP = {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4}
_THAL_MAP = {"normal": 3, "fixed": 6, "reversable": 7}


def _fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        return resp.read()


def _mirror_to_uci(raw: bytes) -> pd.DataFrame:
    """Convert the ISLR re-publication of the Cleveland data to UCI coding."""
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    out = pd.DataFrame({
        "age": df["Age"], "sex": df["Sex"], "cp": df["ChestPain"].map(_CP_MAP),
        "trestbps": df["RestBP"], "chol": df["Chol"], "fbs": df["Fbs"],
        "restecg": df["RestECG"], "thalach": df["MaxHR"], "exang": df["ExAng"],
        "oldpeak": df["Oldpeak"], "slope": df["Slope"], "ca": df["Ca"],
        "thal": df["Thal"].map(_THAL_MAP),
        # Mirror only has Yes/No; UCI 'num' is 0-4. Both binarise the same way.
        "num": (df["AHD"] == "Yes").astype(int),
    })
    return out.reset_index(drop=True)


def download(force: bool = False) -> pd.DataFrame:
    """Download raw data to data/raw (UCI first, mirror as fallback)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_FILE.exists() and not force:
        log.info("Raw file already present: %s", config.RAW_FILE)
        return load_raw()
    try:
        raw = _fetch(config.UCI_URL)
        config.RAW_FILE.write_bytes(raw)
        log.info("Downloaded from UCI -> %s", config.RAW_FILE)
    except Exception as exc:  # network blocked, UCI down, etc.
        log.warning("UCI download failed (%s); using mirror", exc)
        df = _mirror_to_uci(_fetch(config.MIRROR_URL))
        df.to_csv(config.RAW_FILE, header=False, index=False, na_rep="?")
        log.info("Downloaded from mirror -> %s", config.RAW_FILE)
    return load_raw()


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_FILE
    return pd.read_csv(path, header=None, names=config.UCI_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-fix, de-duplicate, binarise the target, sanity-filter values.

    Missing values (ca, thal) are deliberately NOT filled here: imputation
    lives inside the sklearn Pipeline so the same logic runs at serving time.
    """
    df = df.copy()
    df = df.drop_duplicates()
    for col in config.UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physiologically impossible zeros -> missing
    for col in ("trestbps", "chol"):
        df.loc[df[col] <= 0, col] = np.nan

    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df = df.dropna(subset=[config.TARGET])
    return df.reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().loc[lambda s: s > 0]


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Download + clean UCI heart data")
    ap.add_argument("--force", action="store_true", help="re-download even if raw file exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = download(force=args.force)
    df = clean(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Rows=%d  Cols=%d  Positive rate=%.3f", len(df), df.shape[1], df[config.TARGET].mean())
    log.info("Missing values:\n%s", missing_report(df).to_string() or "none")
    log.info("Saved cleaned data -> %s", config.CLEAN_FILE)


if __name__ == "__main__":
    main()
