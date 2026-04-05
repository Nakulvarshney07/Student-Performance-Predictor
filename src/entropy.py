import numpy as np
import pandas as pd


def entropy(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0

    proportions = y.value_counts(normalize=True).values
    proportions = proportions[proportions > 0]

    return -np.sum(proportions * np.log2(proportions))


def information_gain(X_column: pd.Series, y: pd.Series) -> float:
    parent_entropy = entropy(y)
    n = len(y)

    values = X_column.unique()
    weighted_child_entropy = 0.0

    for val in values:
        mask = X_column == val
        subset_y = y[mask]
        weight = len(subset_y) / n
        weighted_child_entropy += weight * entropy(subset_y)

    return parent_entropy - weighted_child_entropy


def split_information(X_column: pd.Series) -> float:
    n = len(X_column)
    proportions = X_column.value_counts(normalize=True).values
    proportions = proportions[proportions > 0]

    return -np.sum(proportions * np.log2(proportions))


def gain_ratio(X_column: pd.Series, y: pd.Series) -> float:
    ig = information_gain(X_column, y)
    si = split_information(X_column)

    if si == 0:
        return 0.0

    return ig / si


def compute_all_gains(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
    y_test = pd.Series([1, 1, 1, 0, 0, 0, 1, 0])
    print(f"Entropy of [1,1,1,0,0,0,1,0] = {entropy(y_test):.4f}")
