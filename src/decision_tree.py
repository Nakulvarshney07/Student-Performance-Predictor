import numpy as np
import pandas as pd
from src.entropy import entropy, gain_ratio


class TreeNode:

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
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}
        self.label = label
        self.is_leaf = is_leaf
        self.info_gain = info_gain
        self.gain_ratio_val = gain_ratio_val


class DecisionTreeC45:

    def __init__(self, max_depth: int = 10, min_samples_split: int = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_importances_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTreeC45":
        self.feature_importances_ = {col: 0.0 for col in X.columns}
        self.tree = self._build_tree(X, y, depth=0)
        total = sum(self.feature_importances_.values())
        if total > 0:
            self.feature_importances_ = {
                k: v / total for k, v in self.feature_importances_.items()
            }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree) for _, row in X.iterrows()])

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        n_samples = len(y)
        n_classes = y.nunique()

        if n_classes == 1:
            return TreeNode(label=y.iloc[0], is_leaf=True)

        if depth >= self.max_depth or n_samples < self.min_samples_split or X.shape[1] == 0:
            majority = y.mode()[0]
            return TreeNode(label=majority, is_leaf=True)

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

        self.feature_importances_[best_feature] += best_gr * n_samples

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
        if node.is_leaf:
            return node.label

        feature_val = row.get(node.feature)

        if feature_val in node.children:
            return self._predict_row(row, node.children[feature_val])
        else:
            child_labels = []
            for child in node.children.values():
                if child.is_leaf:
                    child_labels.append(child.label)
            if child_labels:
                return max(set(child_labels), key=child_labels.count)
            return 1

    def print_tree(self, node=None, indent="", file=None):
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
