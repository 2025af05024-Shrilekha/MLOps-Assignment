import io

import numpy as np
import pandas as pd

from heartml import config, data


def test_clean_binarises_target(raw_df):
    df = data.clean(raw_df)
    assert set(df[config.TARGET].unique()) <= {0, 1}
    assert "num" not in df.columns
    assert ((raw_df["num"] > 0).astype(int).values == df[config.TARGET].values).all()


def test_clean_keeps_missing_for_pipeline(raw_df):
    df = data.clean(raw_df)
    # imputation belongs to the sklearn pipeline, not to cleaning
    assert df["ca"].isna().sum() == 4
    assert df["thal"].isna().sum() == 2


def test_clean_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(10)])
    assert len(data.clean(doubled)) == len(data.clean(raw_df))


def test_clean_zero_cholesterol_becomes_nan(raw_df):
    raw_df.loc[10, "chol"] = 0
    assert np.isnan(data.clean(raw_df).loc[10, "chol"])


def test_load_raw_parses_question_marks(tmp_path):
    line = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
    p = tmp_path / "x.data"
    p.write_text(line)
    df = data.load_raw(p)
    assert list(df.columns) == config.UCI_COLUMNS
    assert pd.isna(df.loc[0, "ca"])


def test_mirror_conversion_matches_uci_coding():
    csv = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
           "1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
           "2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,NA,normal,Yes\n")
    df = data._mirror_to_uci(csv.encode())
    assert list(df.columns) == config.UCI_COLUMNS
    assert df["cp"].tolist() == [1, 4]
    assert df["thal"].tolist() == [6, 3]
    assert df["num"].tolist() == [0, 1]
    assert pd.isna(df.loc[1, "ca"])


def test_mirror_roundtrip_through_csv(tmp_path):
    csv = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
           "1,63,1,typical,145,233,1,2,150,0,2.3,3,NA,fixed,No\n")
    df = data._mirror_to_uci(csv.encode())
    buf = io.StringIO()
    df.to_csv(buf, header=False, index=False, na_rep="?")
    p = tmp_path / "r.data"
    p.write_text(buf.getvalue())
    assert pd.isna(data.load_raw(p).loc[0, "ca"])
