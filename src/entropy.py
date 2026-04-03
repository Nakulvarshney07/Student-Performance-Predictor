"""
entropy.py
==========
Core information-theory functions for decision tree learning.

Implements:
    - Entropy (Shannon entropy)
    - Information Gain (ID3 criterion)
    - Gain Ratio (C4.5 criterion)
    - Split Information

These functions are used by the custom decision tree AND for
mathematical demonstration of how the root node is selected.
"""

import numpy as np
import pandas as pd


def entropy(y: pd.Series) -> float:
    """
    Calculate Shannon entropy of a label series.

    Formula:
        H(S) = -Σ p_i * log2(p_i)

    where p_i is the proportion of class i in the dataset.

    Parameters
    ----------
    y : pd.Series
        Target labels.

    Returns
    -------
    float
        Entropy value (0 = pure, 1 = max impurity for binary).

    Example
    -------
    >>> entropy(pd.Series([1, 1, 0, 0]))
    1.0
    >>> entropy(pd.Series([1, 1, 1, 1]))
    0.0
    """
    if len(y) == 0:
        return 0.0

    proportions = y.value_counts(normalize=True).values
    # Filter out zero proportions to avoid log(0)
    proportions = proportions[proportions > 0]

    return -np.sum(proportions * np.log2(proportions))


def information_gain(X_column: pd.Series, y: pd.Series) -> float:
    """
    Calculate Information Gain for a feature.

    Formula:
        IG(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v)

    where S_v is the subset of S for which attribute A has value v.

    Parameters
    ----------
    X_column : pd.Series
        A single feature column.
    y : pd.Series
        Target labels.

    Returns
    -------
    float
        Information gain value.
    """
    parent_entropy = entropy(y)
    n = len(y)

    # Calculate weighted entropy of children
    values = X_column.unique()
    weighted_child_entropy = 0.0

    for val in values:
        mask = X_column == val
        subset_y = y[mask]
        weight = len(subset_y) / n
        weighted_child_entropy += weight * entropy(subset_y)

    return parent_entropy - weighted_child_entropy


def split_information(X_column: pd.Series) -> float:
    """
    Calculate Split Information (intrinsic value) for Gain Ratio.

    Formula:
        SI(A) = -Σ (|S_v| / |S|) * log2(|S_v| / |S|)

    Parameters
    ----------
    X_column : pd.Series
        A single feature column.

    Returns
    -------
    float
        Split information value.
    """
    n = len(X_column)
    proportions = X_column.value_counts(normalize=True).values
    proportions = proportions[proportions > 0]

    return -np.sum(proportions * np.log2(proportions))


def gain_ratio(X_column: pd.Series, y: pd.Series) -> float:
    """
    Calculate Gain Ratio (C4.5 criterion).

    Formula:
        GR(A) = IG(S, A) / SI(A)

    This normalizes information gain by split information to avoid
    bias toward features with many distinct values.

    Parameters
    ----------
    X_column : pd.Series
        A single feature column.
    y : pd.Series
        Target labels.

    Returns
    -------
    float
        Gain ratio value.
    """
    ig = information_gain(X_column, y)
    si = split_information(X_column)

    if si == 0:
        return 0.0

    return ig / si


def compute_all_gains(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute entropy, information gain, split info, and gain ratio
    for ALL features.  Useful for demonstrating root node selection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.

    Returns
    -------
    pd.DataFrame
        Table with columns: Feature, Info_Gain, Split_Info, Gain_Ratio.
    """
    results = []

    for col in X.columns:
        ig = information_gain(X[col], y)
        si = split_information(X[col])
        gr = gain_ratio(X[col], y)
        results.append({
            "Feature": col,
            "Info_Gain": round(ig, 6),
            "Split_Info": round(si, 6),
            "Gain_Ratio": round(gr, 6),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Gain_Ratio", ascending=False).reset_index(drop=True)

    return results_df


def show_root_node_selection(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Demonstrate mathematically how the root node is selected.

    Returns a formatted string explaining the calculation step by step.
    """
    parent_ent = entropy(y)
    gains_df = compute_all_gains(X, y)
    best_feature = gains_df.iloc[0]["Feature"]

    lines = []
    lines.append("=" * 70)
    lines.append("ROOT NODE SELECTION — Mathematical Demonstration")
    lines.append("=" * 70)
    lines.append(f"\nDataset size: {len(y)}")
    lines.append(f"Pass count: {(y == 1).sum()}, Fail count: {(y == 0).sum()}")
    lines.append(f"\nParent Entropy H(S) = {parent_ent:.6f}")
    lines.append(f"\nFormula: H(S) = -Σ p_i * log2(p_i)")

    p_pass = (y == 1).sum() / len(y)
    p_fail = (y == 0).sum() / len(y)
    lines.append(f"  p(Pass) = {p_pass:.4f}")
    lines.append(f"  p(Fail) = {p_fail:.4f}")
    lines.append(f"  H(S) = -({p_pass:.4f} * log2({p_pass:.4f})) - ({p_fail:.4f} * log2({p_fail:.4f}))")
    lines.append(f"  H(S) = {parent_ent:.6f}")

    lines.append(f"\n{'Feature':<20} {'Info Gain':<15} {'Split Info':<15} {'Gain Ratio':<15}")
    lines.append("-" * 65)

    for _, row in gains_df.head(10).iterrows():
        lines.append(f"{row['Feature']:<20} {row['Info_Gain']:<15.6f} {row['Split_Info']:<15.6f} {row['Gain_Ratio']:<15.6f}")

    lines.append(f"\n>>> Best feature for root node (highest Gain Ratio): {best_feature}")
    lines.append(f"    Gain Ratio = {gains_df.iloc[0]['Gain_Ratio']:.6f}")
    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    y_test = pd.Series([1, 1, 1, 0, 0, 0, 1, 0])
    print(f"Entropy of [1,1,1,0,0,0,1,0] = {entropy(y_test):.4f}")  # Should be 1.0
