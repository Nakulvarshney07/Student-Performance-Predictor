import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def visualize_tree(
    sklearn_tree,
    feature_names: list,
    output_dir: str = "outputs",
    max_depth: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(28, 14))
    plot_tree(
        sklearn_tree,
        feature_names=feature_names,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
    )
    ax.set_title(
        "Decision Tree — Student Performance Predictor (ID3/C4.5 with Entropy)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    filepath = os.path.join(output_dir, "tree.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualize] Decision tree saved to {filepath}")


def visualize_feature_importance(
    feature_importances: dict,
    title: str = "Feature Importance",
    output_dir: str = "outputs",
    filename: str = "feature_importance.png",
):
    os.makedirs(output_dir, exist_ok=True)

    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    bars = ax.barh(features, importances, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, importances):
        if val > 0.01:
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualize] Feature importance chart saved to {filepath}")


def visualize_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    output_dir: str = "outputs",
    filename: str = "confusion_matrix.png",
):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"],
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")

    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualize] Confusion matrix saved to {filepath}")


def visualize_pipeline(results: dict, output_dir: str = "outputs"):
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    visualize_tree(
        results["sklearn_tree"],
        results["feature_names"],
        output_dir=output_dir,
    )

    sklearn_importances = dict(
        zip(results["feature_names"], results["sklearn_tree"].feature_importances_)
    )
    visualize_feature_importance(
        sklearn_importances,
        title="Feature Importance — sklearn DecisionTree (Entropy)",
        output_dir=output_dir,
        filename="feature_importance.png",
    )

    sklearn_preds = results["sklearn_tree"].predict(results["X_test"])
    visualize_confusion_matrix(
        results["y_test"],
        sklearn_preds,
        title="Confusion Matrix — sklearn DecisionTree",
        output_dir=output_dir,
        filename="confusion_matrix.png",
    )

    print(f"\n[Visualize] All visualizations saved to '{output_dir}/' directory")


if __name__ == "__main__":
    print("Run from main.py to generate visualizations.")
