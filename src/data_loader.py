import os
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, sep=";")
    print(f"[DataLoader] Loaded {filepath} — shape: {df.shape}")
    return df


def load_and_merge(data_dir: str = "data") -> pd.DataFrame:
    mat_path = os.path.join(data_dir, "student-mat.csv")
    por_path = os.path.join(data_dir, "student-por.csv")

    df_mat = load_dataset(mat_path)
    df_por = load_dataset(por_path)

    df_mat["subject"] = "Math"
    df_por["subject"] = "Portuguese"

    df_combined = pd.concat([df_mat, df_por], ignore_index=True)

    print(f"[DataLoader] Combined dataset shape: {df_combined.shape}")
    print(f"[DataLoader] Columns: {list(df_combined.columns)}")

    return df_combined


if __name__ == "__main__":
    df = load_and_merge()
    print(df.head())
    print(f"\nTotal records: {len(df)}")
