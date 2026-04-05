import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=["Fail", "Pass"], zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }

    return metrics


def print_metrics(metrics: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"EVALUATION RESULTS — {metrics['model_name']}")
    lines.append("=" * 60)
    lines.append(f"\n  Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall:    {metrics['recall']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1_score']:.4f}")

    lines.append(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    lines.append(f"                Predicted Fail  Predicted Pass")
    lines.append(f"  Actual Fail   {cm[0][0]:>10}       {cm[0][1]:>10}")
    lines.append(f"  Actual Pass   {cm[1][0]:>10}       {cm[1][1]:>10}")

    lines.append(f"\n  Classification Report:")
    lines.append(metrics["classification_report"])
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)
    return output


def print_feature_importance(
    feature_importances: dict,
    title: str = "Feature Importance",
) -> str:
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )

    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"{title}")
    lines.append(f"{'=' * 60}")
    lines.append(f"{'Rank':<6} {'Feature':<25} {'Importance':<15}")
    lines.append("-" * 50)

    for i, (feat, imp) in enumerate(sorted_features, 1):
        bar = "█" * int(imp * 40)
        lines.append(f"{i:<6} {feat:<25} {imp:<15.6f} {bar}")

    output = "\n".join(lines)
    print(output)
    return output


def save_metrics(metrics_text: str, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "metrics.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"\n[Evaluate] Metrics saved to {filepath}")


def evaluate_pipeline(results: dict, output_dir: str = "outputs") -> str:
    all_text = []

    print("\n[Evaluating Custom C4.5 Decision Tree]")
    custom_preds = results["custom_tree"].predict(results["X_test"])
    custom_metrics = evaluate_model(results["y_test"], custom_preds, "Custom C4.5 Decision Tree")
    text1 = print_metrics(custom_metrics)
    all_text.append(text1)

    imp_text1 = print_feature_importance(
        results["custom_tree"].feature_importances_,
        "Feature Importance — Custom C4.5 Tree",
    )
    all_text.append(imp_text1)

    print("\n[Evaluating sklearn DecisionTreeClassifier]")
    sklearn_preds = results["sklearn_tree"].predict(results["X_test"])
    sklearn_metrics = evaluate_model(results["y_test"], sklearn_preds, "sklearn DecisionTreeClassifier")
    text2 = print_metrics(sklearn_metrics)
    all_text.append(text2)

    sklearn_importances = dict(
        zip(results["feature_names"], results["sklearn_tree"].feature_importances_)
    )
    imp_text2 = print_feature_importance(
        sklearn_importances,
        "Feature Importance — sklearn DecisionTree",
    )
    all_text.append(imp_text2)

    all_text.insert(0, results.get("root_analysis", ""))

    combined = "\n\n".join(all_text)
    save_metrics(combined, output_dir)

    return combined


if __name__ == "__main__":
    print("Run from main.py to see evaluation results.")
