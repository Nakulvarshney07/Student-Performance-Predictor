"""
decision_tree.py
================
Custom Decision Tree implementation using the C4.5 algorithm
(Gain Ratio as splitting criterion).

Features:
    - Builds a tree recursively using gain ratio
    - Handles both categorical and numerical features
    - Supports max_depth and min_samples_split stopping criteria
    - Provides a text-based tree representation

For comparison, we also wrap sklearn's DecisionTreeClassifier to enable
proper visualization via plot_tree.
"""

import numpy as np
import pandas as pd
from src.entropy import entropy, gain_ratio


# ──────────────────────────────────────────────
# Custom C4.5 Decision Tree (from scratch)
# ──────────────────────────────────────────────

class TreeNode:
    """Represents a node in the decision tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        children=None,
        label=None,
        is_leaf=False,
        info_gain=0.0,
        gain_ratio_val=0.0,
    ):
        self.feature = feature            # Feature name used for split
        self.threshold = threshold        # Threshold for numeric split
        self.children = children or {}    # {value: TreeNode} for categorical
        self.label = label                # Class label (for leaf nodes)
        self.is_leaf = is_leaf
        self.info_gain = info_gain
        self.gain_ratio_val = gain_ratio_val


class DecisionTreeC45:
    """
    C4.5 Decision Tree Classifier built from scratch.

    Uses Gain Ratio for splitting to avoid bias toward
    high-cardinality features.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 5):
        """
        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree.
        min_samples_split : int
            Minimum samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_importances_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTreeC45":
        """Build the decision tree."""
        self.feature_importances_ = {col: 0.0 for col in X.columns}
        self.tree = self._build_tree(X, y, depth=0)
        # Normalize feature importances
        total = sum(self.feature_importances_.values())
        if total > 0:
            self.feature_importances_ = {
                k: v / total for k, v in self.feature_importances_.items()
            }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for samples in X."""
        return np.array([self._predict_row(row, self.tree) for _, row in X.iterrows()])

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        """Recursively build the tree."""
        n_samples = len(y)
        n_classes = y.nunique()

        # ── Stopping conditions ──
        if n_classes == 1:
            return TreeNode(label=y.iloc[0], is_leaf=True)

        if depth >= self.max_depth or n_samples < self.min_samples_split or X.shape[1] == 0:
            majority = y.mode()[0]
            return TreeNode(label=majority, is_leaf=True)

        # ── Find best split (highest gain ratio) ──
        best_feature = None
        best_gr = -1

        for col in X.columns:
            gr = gain_ratio(X[col], y)
            if gr > best_gr:
                best_gr = gr
                best_feature = col

        if best_feature is None or best_gr <= 0:
            majority = y.mode()[0]
            return TreeNode(label=majority, is_leaf=True)

        # Track feature importance
        self.feature_importances_[best_feature] += best_gr * n_samples

        # ── Split on best feature ──
        node = TreeNode(
            feature=best_feature,
            gain_ratio_val=best_gr,
        )

        unique_values = X[best_feature].unique()

        for val in unique_values:
            mask = X[best_feature] == val
            subset_X = X[mask].drop(columns=[best_feature])
            subset_y = y[mask]

            if len(subset_y) == 0:
                child = TreeNode(label=y.mode()[0], is_leaf=True)
            else:
                child = self._build_tree(subset_X, subset_y, depth + 1)

            node.children[val] = child

        return node

    def _predict_row(self, row: pd.Series, node: TreeNode):
        """Predict a single sample by traversing the tree."""
        if node.is_leaf:
            return node.label

        feature_val = row.get(node.feature)

        if feature_val in node.children:
            return self._predict_row(row, node.children[feature_val])
        else:
            # Unseen value — return most common child label
            child_labels = []
            for child in node.children.values():
                if child.is_leaf:
                    child_labels.append(child.label)
            if child_labels:
                return max(set(child_labels), key=child_labels.count)
            return 1  # default to Pass

    def print_tree(self, node=None, indent="", file=None):
        """Print a text representation of the tree."""
        if node is None:
            node = self.tree

        if node.is_leaf:
            label_str = "Pass" if node.label == 1 else "Fail"
            line = f"{indent}→ [{label_str}]"
            print(line) if file is None else print(line, file=file)
            return

        line = f"{indent}[{node.feature}] (GR={node.gain_ratio_val:.4f})"
        print(line) if file is None else print(line, file=file)

        for val, child in node.children.items():
            line = f"{indent}  ├── {node.feature} == {val}:"
            print(line) if file is None else print(line, file=file)
            self.print_tree(child, indent + "  │   ", file=file)


if __name__ == "__main__":
    # Quick test with a small dataset
    data = {
        "Outlook": [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2],
        "Temp": [2, 2, 2, 1, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1],
        "Play": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    }
    df = pd.DataFrame(data)
    X = df[["Outlook", "Temp"]]
    y = df["Play"]

    tree = DecisionTreeC45(max_depth=5)
    tree.fit(X, y)
    tree.print_tree()
    print(f"\nPredictions: {tree.predict(X)}")
