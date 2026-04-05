import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_merge
from src.preprocessing import preprocess_pipeline
from src.decision_tree import DecisionTreeC45
from src.entropy import show_root_node_selection


def train_pipeline(
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 8,
) -> dict:
    print("\n" + "=" * 60)
    print("STUDENT PERFORMANCE PREDICTOR — TRAINING PIPELINE")
    print("=" * 60)

    print("\n[STEP 1] Loading datasets...")
    df = load_and_merge(data_dir)

    print("\n[STEP 2] Preprocessing data...")
    X, y, encoders = preprocess_pipeline(df)

    print("\n[STEP 3] Root Node Selection Analysis...")
    root_analysis = show_root_node_selection(X, y)
    print(root_analysis)

    print("\n[STEP 4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    print("\n[STEP 5a] Training Custom C4.5 Decision Tree...")
    custom_tree = DecisionTreeC45(max_depth=max_depth, min_samples_split=5)
    custom_tree.fit(X_train, y_train)
    print("  ✓ Custom C4.5 tree trained successfully")

    print("\n[STEP 5b] Training sklearn DecisionTreeClassifier...")
    sklearn_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_split=5,
        random_state=random_state,
    )
    sklearn_tree.fit(X_train, y_train)
    print("  ✓ sklearn Decision Tree trained successfully")

    results = {
        "custom_tree": custom_tree,
        "sklearn_tree": sklearn_tree,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "encoders": encoders,
        "root_analysis": root_analysis,
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = train_pipeline()
    print(f"\nCustom tree feature importances (top 5):")
    importances = sorted(
        results["custom_tree"].feature_importances_.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for feat, imp in importances[:5]:
        print(f"  {feat}: {imp:.4f}")
