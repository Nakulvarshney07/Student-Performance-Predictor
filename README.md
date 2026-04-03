# 🎓 Student Performance Predictor using Decision Tree (ID3 / C4.5)

A **Data Mining** project that predicts whether a student will **Pass or Fail** based on demographic, social, and academic attributes using **Decision Tree classification** (ID3 / C4.5 algorithm).

---

## 📁 Project Structure

```
Student-Performance-Predictor/
│
├── data/
│   ├── student-mat.csv          # Math course dataset (396 students)
│   └── student-por.csv          # Portuguese course dataset (650 students)
│
├── src/
│   ├── __init__.py              # Package init
│   ├── data_loader.py           # Load + merge datasets
│   ├── preprocessing.py         # Cleaning, encoding, feature selection
│   ├── entropy.py               # Entropy, Information Gain, Gain Ratio
│   ├── decision_tree.py         # Custom C4.5 implementation from scratch
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Metrics + confusion matrix
│   └── visualize.py             # Tree & chart visualization
│
├── outputs/
│   ├── tree.png                 # Decision tree visualization
│   ├── feature_importance.png   # Feature importance bar chart
│   ├── confusion_matrix.png     # Confusion matrix heatmap
│   └── metrics.txt              # All evaluation metrics
│
├── main.py                      # Runs entire pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline:
```bash
python main.py
```

This will:
- Load both datasets (`student-mat.csv`, `student-por.csv`)
- Preprocess and create binary target (Pass/Fail)
- Show mathematical root node selection
- Train both a **custom C4.5 tree** and **sklearn DecisionTree**
- Print evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Generate visualizations in `outputs/`

---

## 📚 Dataset Information

**Source:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

**Attributes (33 columns):**

| Category | Features |
|----------|----------|
| **Demographic** | school, sex, age, address, famsize, Pstatus |
| **Family** | Medu, Fedu, Mjob, Fjob, guardian |
| **Education** | reason, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet |
| **Social** | romantic, famrel, freetime, goout, Dalc, Walc, health, absences |
| **Grades** | G1 (period 1), G2 (period 2), **G3 (final grade — target)** |

**Target Variable:**
- `G3 >= 10` → **Pass** (1)
- `G3 < 10` → **Fail** (0)

> Note: G1 and G2 are **dropped** during training to prevent data leakage — the model predicts from demographic/behavioural features only.

---

## 🧮 Key Concepts Explained

### 1. Entropy (Shannon Entropy)

Entropy measures the **impurity** or **uncertainty** in a dataset. It tells us how mixed the classes are.

**Formula:**

```
H(S) = -Σ p_i × log₂(p_i)
```

Where `p_i` is the proportion of class `i` in the dataset.

**Examples:**
- If all students Pass → H(S) = 0 (pure, no uncertainty)
- If 50% Pass, 50% Fail → H(S) = 1.0 (maximum uncertainty)
- If 75% Pass, 25% Fail → H(S) = 0.811

**Interpretation:** Lower entropy = more pure subset = better for classification.

---

### 2. Information Gain (ID3 Criterion)

Information Gain measures how much a feature **reduces entropy** when used to split the data.

**Formula:**

```
IG(S, A) = H(S) - Σ (|S_v| / |S|) × H(S_v)
```

Where:
- `H(S)` = entropy of the original dataset
- `S_v` = subset where attribute A has value v
- `|S_v| / |S|` = proportion of samples in subset v

**How root node is selected:** The feature with the **highest Information Gain** is chosen as the root node in the **ID3 algorithm**.

**Limitation:** ID3's Information Gain is biased toward attributes with many distinct values (e.g., an ID column would have perfect gain but is useless).

---

### 3. Gain Ratio (C4.5 Improvement)

C4.5 improves upon ID3 by normalizing Information Gain with **Split Information** to handle the multi-value bias.

**Split Information:**
```
SI(A) = -Σ (|S_v| / |S|) × log₂(|S_v| / |S|)
```

**Gain Ratio:**
```
GR(A) = IG(S, A) / SI(A)
```

The feature with the **highest Gain Ratio** is selected for splitting. This prevents features with many categories from being unfairly favored.

---

### 4. How Decision Trees Work

1. **Start** with the full dataset at the root node
2. **Calculate** Gain Ratio for every feature
3. **Select** the feature with the highest Gain Ratio → becomes the splitting attribute
4. **Create branches** for each value of the selected feature
5. **Repeat** recursively for each branch (subset of data)
6. **Stop** when:
   - All samples in a node belong to one class (pure)
   - Maximum depth is reached
   - Minimum samples threshold is met

```
                    [Root: Best Feature]
                   /         |          \
            [Value 1]   [Value 2]   [Value 3]
              /              |              \
         [Next Best]     [PASS]        [Next Best]
           /    \                         /    \
       [FAIL]  [PASS]               [FAIL]  [PASS]
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Of all predicted Pass, how many actually passed? |
| **Recall** | Of all actual Pass students, how many did we correctly identify? |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | 2×2 table showing TP, TN, FP, FN |

---

## 🛠️ Implementation Details

### Two Models Trained:

1. **Custom C4.5 Tree (from scratch)**
   - Uses `entropy.py` functions for gain ratio calculation
   - Implements recursive tree building in `decision_tree.py`
   - Shows the mathematical process of node selection

2. **sklearn DecisionTreeClassifier**
   - Uses `criterion="entropy"` (equivalent to ID3)
   - Used primarily for tree visualization (`plot_tree`)
   - Provides a benchmark for the custom implementation

### Why Both?
The custom implementation demonstrates **understanding of the algorithm**, while sklearn provides **professional-grade visualization** and serves as a correctness benchmark.

---

## 📄 Output Files

After running `python main.py`, check the `outputs/` folder:

- **`metrics.txt`** — Complete evaluation report with mathematical root node selection
- **`tree.png`** — Visual decision tree diagram
- **`feature_importance.png`** — Bar chart of feature importances
- **`confusion_matrix.png`** — Heatmap of the confusion matrix

---

## 👨‍💻 Author

Data Mining Course Project — Student Performance Prediction using Decision Trees

---

## 📖 References

1. P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Student+Performance
3. Quinlan, J.R. *C4.5: Programs for Machine Learning.* Morgan Kaufmann, 1993.
