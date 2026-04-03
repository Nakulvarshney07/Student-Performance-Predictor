"""
main.py
=======
Entry point for the Student Performance Predictor project.

Runs the complete pipeline:
    1. Load & merge datasets
    2. Preprocess data
    3. Train Decision Trees (Custom C4.5 + sklearn)
    4. Evaluate models
    5. Generate visualizations
    6. Save all outputs

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_pipeline
from src.evaluate import evaluate_pipeline
from src.visualize import visualize_pipeline


def main():
    """Run the entire Student Performance Predictor pipeline."""

    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║   Student Performance Predictor                        ║
    ║   Decision Tree (ID3 / C4.5) — Data Mining Project     ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # ── Configuration ──
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_DEPTH = 8

    # ── Step 1 & 2 & 3: Train ──
    results = train_pipeline(
        data_dir=DATA_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
    )

    # ── Step 4: Evaluate ──
    evaluate_pipeline(results, output_dir=OUTPUT_DIR)

    # ── Step 5: Visualize ──
    visualize_pipeline(results, output_dir=OUTPUT_DIR)

    # ── Step 6: Print custom tree structure ──
    print("\n" + "=" * 60)
    print("CUSTOM C4.5 DECISION TREE STRUCTURE (first few levels)")
    print("=" * 60)
    results["custom_tree"].print_tree()

    # ── Summary ──
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