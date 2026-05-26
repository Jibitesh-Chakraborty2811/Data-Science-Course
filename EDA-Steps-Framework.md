# A Practical EDA Framework Used by Data Scientists

This is a reusable framework you can apply to almost any dataset.

---

# COMPLETE EDA FLOW

```text
1. Understand business problem
2. Understand columns and data types
3. Handle missing values
4. Analyze distributions
5. Detect and handle outliers
6. Analyze relationships with target
7. Feature importance analysis
8. Handle correlations/redundancy
9. Feature engineering
10. Final preprocessing decisions
```

---

# STEP 0 — Understand Data Types

Before ANYTHING:

```python
df.info()
df.describe()
```

Split columns:

```python
numerical_cols = [...]
categorical_cols = [...]
target_col = "target"
```

---

# STEP 1 — HANDLE MISSING VALUES

This is ALWAYS one of the first steps.

---

# PART A — NUMERICAL MISSING VALUES FRAMEWORK

---

# Step 1 — Find Missing Percentage

```python
missing_percent = df[col].isnull().mean() * 100
```

---

# Step 2 — Analyze Distribution

Check:

* Histogram
* Boxplot
* Skewness

```python
sns.histplot(df[col])
sns.boxplot(x=df[col])

df[col].skew()
```

---

# Step 3 — Decide Missing Value Strategy

---

# CASE 1 — Missing Values SMALL (< 5%)

Usually safe to impute.

---

## If distribution is NORMAL

Conditions:

```python
abs(skewness) < 0.5
```

Use:

* Mean imputation

```python
df[col].fillna(df[col].mean())
```

Why?

* Mean represents center well in symmetric data.

---

## If distribution is SKEWED

Conditions:

```python
abs(skewness) > 0.5
```

Use:

* Median imputation

```python
df[col].fillna(df[col].median())
```

Why?

* Median resistant to outliers.

---

# CASE 2 — Missing Values MODERATE (5%–30%)

Now think carefully.

---

## Option A — Median/Mean Imputation

Still okay if:

* Feature important
* Missingness random

---

## Option B — Predictive Imputation

Use ML models:

* KNN Imputer
* Iterative Imputer

Used when:

* Column highly important
* Strong feature relationships exist

---

## Option C — Add Missing Indicator

```python
df[col+"_missing"] = df[col].isnull().astype(int)
```

Sometimes missing itself carries information.

Example:

* Missing income may indicate unemployed customers.

---

# CASE 3 — Missing Values VERY HIGH (> 40–60%)

Ask:

* Is feature business critical?

If NO:

* Drop column

If YES:

* Advanced imputation
* Domain knowledge

---

# NUMERICAL MISSING VALUE DECISION TREE

```text
Missing values?
    |
    ├── <5%
    |      |
    |      ├── Normal → Mean
    |      └── Skewed → Median
    |
    ├── 5–30%
    |      |
    |      ├── Important feature → Advanced imputation
    |      └── Less important → Median/Mean
    |
    └── >40–60%
           |
           ├── Critical feature → Domain/ML imputation
           └── Otherwise → Drop
```

---

# PART B — CATEGORICAL MISSING VALUES FRAMEWORK

---

# Step 1 — Count Missing %

```python
df[col].isnull().mean()
```

---

# Step 2 — Analyze Cardinality

Check:

```python
df[col].nunique()
```

---

# Step 3 — Decide Strategy

---

# CASE 1 — Small Missing %

Use:

* Mode imputation

```python
df[col].fillna(df[col].mode()[0])
```

Best for:

* Low missing %
* Stable categories

---

# CASE 2 — Missing May Have Meaning

Create separate category:

```python
df[col].fillna("Unknown")
```

Example:

* Occupation missing
* Education missing

Very common in industry.

---

# CASE 3 — High Missing %

If:

* Too sparse
* Too many unique values
* Low business value

Drop feature.

---

# CATEGORICAL MISSING VALUE DECISION TREE

```text
Missing values?
    |
    ├── Small %
    |      └── Mode
    |
    ├── Missing has business meaning
    |      └── "Unknown"
    |
    └── Very high missing
           └── Drop feature
```

---

# STEP 2 — OUTLIER DETECTION FRAMEWORK

ONLY for numerical columns.

Categorical columns technically do not have outliers.

(Though rare categories can exist.)

---

# PART A — UNDERSTAND DISTRIBUTION FIRST

This is CRITICAL.

---

# Step 1 — Visualize Distribution

```python
sns.histplot(df[col], kde=True)
sns.boxplot(x=df[col])
```

---

# Step 2 — Check Skewness

```python
df[col].skew()
```

Interpretation:

| Skewness | Meaning    |
| -------- | ---------- |
| 0        | Symmetric  |
| > 0      | Right skew |
| < 0      | Left skew  |

Rules:

| Value       | Interpretation       |
| ----------- | -------------------- |
| -0.5 to 0.5 | Approximately normal |
| 0.5 to 1    | Moderately skewed    |
| >1          | Highly skewed        |

---

# Step 3 — Check Kurtosis

```python
df[col].kurtosis()
```

Measures:

* Tail heaviness

Interpretation:

| Kurtosis | Meaning              |
| -------- | -------------------- |
| ~0       | Normal               |
| High     | Heavy tails/outliers |
| Low      | Flat distribution    |

---

# PART B — CHOOSE OUTLIER METHOD

---

# CASE 1 — Approximately Normal Distribution

Conditions:

```python
abs(skewness) < 0.5
```

Use:

* Z-score

---

# Z-Score Method

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"}}

```python
from scipy.stats import zscore

z = np.abs(zscore(df[col]))

outliers = df[z > 3]
```

Use when:

* Bell-shaped distribution
* Gaussian-like data

---

# CASE 2 — Skewed Distribution

Conditions:

```python
abs(skewness) > 0.5
```

Use:

* IQR

---

# IQR Method

IQR = Q_3 - Q_1

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Outliers:

```python
df[(df[col] < lower) | (df[col] > upper)]
```

---

# OUTLIER DETECTION FLOW

```text
Check distribution
    |
    ├── Approximately normal
    |      └── Z-score
    |
    └── Skewed / non-normal
           └── IQR
```

---

# PART C — OUTLIER HANDLING

---

# Option 1 — Remove

Use when:

* Data clearly invalid

Example:

* Age = 500

---

# Option 2 — Capping (Winsorization)

Replace extremes.

```python
df[col] = np.clip(df[col], lower, upper)
```

Most common industry solution.

---

# Option 3 — Log Transform

For right-skewed data.

```python
df[col] = np.log1p(df[col])
```

Reduces skewness.

---

# Option 4 — Keep Outliers

Sometimes outliers are valuable.

Example:

* Fraud detection
* Financial spikes

Never blindly remove outliers.

---

# CATEGORICAL "OUTLIERS"

Not true outliers, but:

* Rare categories
* High cardinality issues

---

# Detect Rare Categories

```python
df[col].value_counts(normalize=True)
```

Example:

| Category | %    |
| -------- | ---- |
| A        | 80%  |
| B        | 18%  |
| C        | 0.2% |

Rare categories may be grouped:

```python
Other
```

---

# STEP 3 — FEATURE IMPORTANCE FRAMEWORK

Now:

> Which features affect target?

This depends on:

* Feature type
* Target type

---

# CASE A — NUMERICAL FEATURE vs NUMERICAL TARGET

Example:

* Income vs Sales

---

# Methods

## Correlation

```python
df.corr()
```

Visualization:

```python
sns.heatmap(df.corr(), annot=True)
```

Use:

* Linear relationships

---

## Scatterplot

```python
sns.scatterplot(x=feature, y=target)
```

---

# CASE B — NUMERICAL FEATURE vs CATEGORICAL TARGET

VERY common classification scenario.

Example:

* Salary vs Churn

---

# Visualizations

## Boxplot

```python
sns.boxplot(x=target, y=feature)
```

---

# Statistical Test

## T-test (binary target)

```python
ttest_ind()
```

## ANOVA (multiple classes)

```python
f_oneway()
```

---

# ML-Based Importance

## Mutual Information

```python
mutual_info_classif()
```

## Random Forest Importance

```python
model.feature_importances_
```

Very common in industry.

---

# CASE C — CATEGORICAL FEATURE vs CATEGORICAL TARGET

Example:

* Gender vs Churn

---

# Visualization

```python
sns.countplot(x=feature, hue=target)
```

---

# Statistical Test

## Chi-Square Test

Used for:

* Association between categories

```python
chi2_contingency()
```

---

# CASE D — CATEGORICAL FEATURE vs NUMERICAL TARGET

Example:

* City vs Revenue

---

# Visualization

```python
sns.boxplot(x=feature, y=target)
```

---

# Statistical Test

## ANOVA

Checks:

* Do category groups differ significantly?

---

# FEATURE IMPORTANCE MASTER TABLE

| Feature Type | Target Type | Visualization | Statistical Method |
| ------------ | ----------- | ------------- | ------------------ |
| Numeric      | Numeric     | Scatterplot   | Correlation        |
| Numeric      | Categorical | Boxplot       | T-test / ANOVA     |
| Categorical  | Categorical | Countplot     | Chi-square         |
| Categorical  | Numeric     | Boxplot       | ANOVA              |

---

# REAL INDUSTRY FEATURE IMPORTANCE FLOW

```text
1. Understand target type
2. Analyze feature-target relationship visually
3. Apply statistical significance test
4. Apply ML-based feature importance
5. Check business meaning
6. Remove redundant/useless features
```

---

# GOLDEN RULES OF EDA

---

# Rule 1

Never blindly apply techniques.

Always:

* Understand distribution first.

---

# Rule 2

Visualization before statistics.

Humans detect patterns visually faster.

---

# Rule 3

Business understanding matters more than formulas.

---

# Rule 4

Outliers are not always bad.

---

# Rule 5

Correlation does NOT imply causation.

---

# Rule 6

Missing data itself may contain information.

---

# FINAL UNIVERSAL EDA CHECKLIST

```text
1. Understand business problem
2. Identify target variable
3. Identify numerical/categorical columns
4. Analyze missing values
5. Handle missing values
6. Analyze distributions
7. Detect skewness/kurtosis
8. Detect outliers
9. Handle outliers
10. Analyze feature-target relationships
11. Statistical significance testing
12. Correlation analysis
13. Feature importance analysis
14. Remove redundancy
15. Feature engineering
16. Prepare for ML
```
