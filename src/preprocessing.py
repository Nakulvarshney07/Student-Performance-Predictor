"""
preprocessing.py
=================
Handles data cleaning, encoding of categorical features,
binary target creation (Pass/Fail), and feature selection.

Target variable:
    - G3 >= 10 → Pass (1)
    - G3 < 10  → Fail (0)
    (Based on Portuguese grading system where 10 is the pass mark.)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_target(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Create a binary target column 'Pass_Fail' from the final grade G3.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with column 'G3'.
    threshold : int
        Minimum grade to pass (default 10).

    Returns
    -------
    pd.DataFrame
        Dataframe with new 'Pass_Fail' column (1=Pass, 0=Fail).
    """
    df = df.copy()
    df["Pass_Fail"] = (df["G3"] >= threshold).astype(int)

    pass_count = df["Pass_Fail"].sum()
    fail_count = len(df) - pass_count
    print(f"[Preprocessing] Target created — Pass: {pass_count}, Fail: {fail_count}")

    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical (object) columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Encoded dataframe and dict of {column: LabelEncoder}.
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"[Preprocessing] Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select feature columns (X) and target (y).

    Drops G1, G2 (intermediate grades) and G3 (raw grade) to prevent
    data leakage — the model should predict Pass/Fail from demographic
    and behavioural features only.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'Pass_Fail' column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) — features and target.
    """
    # Columns to drop: intermediate grades + raw target + subject tag
    drop_cols = ["G1", "G2", "G3", "Pass_Fail"]

    # Also drop 'subject' if it exists (it's a synthetic column)
    if "subject" in df.columns:
        drop_cols.append("subject")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Pass_Fail"]

    print(f"[Preprocessing] Features shape: {X.shape}")
    print(f"[Preprocessing] Feature list: {list(X.columns)}")

    return X, y


def preprocess_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full preprocessing pipeline.

    Steps:
    1. Create binary target (Pass/Fail)
    2. Label-encode categorical columns
    3. Select features and target

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from data_loader.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict]
        (X, y, encoders)
    """
    df = create_target(df)
    df, encoders = encode_categorical(df)
    X, y = select_features(df)

    return X, y, encoders


if __name__ == "__main__":
    from data_loader import load_and_merge

    df = load_and_merge()
    X, y, enc = preprocess_pipeline(df)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"y distribution:\n{y.value_counts()}")
