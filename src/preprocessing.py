import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_target(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["Pass_Fail"] = (df["G3"] >= threshold).astype(int)

    pass_count = df["Pass_Fail"].sum()
    fail_count = len(df) - pass_count
    print(f"[Preprocessing] Target created — Pass: {pass_count}, Fail: {fail_count}")

    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    drop_cols = ["G1", "G2", "G3", "Pass_Fail"]

    if "subject" in df.columns:
        drop_cols.append("subject")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Pass_Fail"]

    print(f"[Preprocessing] Features shape: {X.shape}")
    print(f"[Preprocessing] Feature list: {list(X.columns)}")

    return X, y


def preprocess_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
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
