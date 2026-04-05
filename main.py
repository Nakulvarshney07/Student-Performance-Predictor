import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_pipeline
from src.evaluate import evaluate_pipeline
from src.visualize import visualize_pipeline


def main():


    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_DEPTH = 8

    results = train_pipeline(
        data_dir=DATA_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
    )

    evaluate_pipeline(results, output_dir=OUTPUT_DIR)

    visualize_pipeline(results, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("CUSTOM C4.5 DECISION TREE STRUCTURE (first few levels)")
    print("=" * 60)
    results["custom_tree"].print_tree()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Output Files")
    print("=" * 60)
    print(f"  📊 outputs/metrics.txt           — Evaluation metrics")
    print(f"  🌳 outputs/tree.png              — Decision tree visualization")
    print(f"  📈 outputs/feature_importance.png — Feature importance chart")
    print(f"  🔲 outputs/confusion_matrix.png  — Confusion matrix heatmap")
    print("=" * 60)


if __name__ == "__main__":
    main()