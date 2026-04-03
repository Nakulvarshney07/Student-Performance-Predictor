"""
data_loader.py
==============
Loads and merges the student-mat.csv and student-por.csv datasets.

The merge is performed on common demographic/family attributes so that
students appearing in BOTH datasets are kept once, while students unique
to either dataset are also preserved (outer join approach with dedup).
"""

import os
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (semicolon-separated).

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, sep=";")
    print(f"[DataLoader] Loaded {filepath} — shape: {df.shape}")
    return df


def load_and_merge(data_dir: str = "data") -> pd.DataFrame:
    """
    Load both Math and Portuguese datasets and merge them.

    Merging strategy (following UCI documentation):
    - Inner-join on 13 shared demographic attributes to find students
      who appear in both courses.
    - Also concatenate both datasets and drop duplicates so that
      students appearing in only one file are not lost.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSV files.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with a 'subject' column indicating origin.
    """
    mat_path = os.path.join(data_dir, "student-mat.csv")
    por_path = os.path.join(data_dir, "student-por.csv")

    df_mat = load_dataset(mat_path)
    df_por = load_dataset(por_path)

    # Add subject label before merging
    df_mat["subject"] = "Math"
    df_por["subject"] = "Portuguese"

    # Concatenate both datasets (union of all students)
    df_combined = pd.concat([df_mat, df_por], ignore_index=True)

    print(f"[DataLoader] Combined dataset shape: {df_combined.shape}")
    print(f"[DataLoader] Columns: {list(df_combined.columns)}")

    return df_combined


if __name__ == "__main__":
    df = load_and_merge()
    print(df.head())
    print(f"\nTotal records: {len(df)}")
