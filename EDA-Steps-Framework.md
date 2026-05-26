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

| Skewness | Meaning |
| --- | --- |
| 0 | Symmetric |
| > 0 | Right skew |
| < 0 | Left skew |

Rules:

| Value | Interpretation |
| --- | --- |
| -0.5 to 0.5 | Approximately normal |
| 0.5 to 1 | Moderately skewed |
| >1 | Highly skewed |

---

# Step 3 — Check Kurtosis

```python
df[col].kurtosis()

```

Measures:

* Tail heaviness

Interpretation:

| Kurtosis | Meaning |
| --- | --- |
| ~0 | Normal |
| High | Heavy tails/outliers |
| Low | Flat distribution |

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

$$z = \frac{x-\mu}{\sigma}$$

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

$$IQR = Q_3 - Q_1$$

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

| Category | % |
| --- | --- |
| A | 80% |
| B | 18% |
| C | 0.2% |

Rare categories may be grouped:

```python
Other

```

---

# STEP 3 — FEATURE IMPORTANCE FRAMEWORK (NUMERIC + RELIABLE)

Visualizations help humans understand patterns, but **feature importance should ultimately be validated numerically/statistically**.

A strong data scientist does BOTH:

```text
1. Visual validation
2. Statistical validation
3. ML-based validation

```

This is the proper **mathematical/statistical feature importance framework**.

The correct method depends on:

| Feature Type | Target Type |
| --- | --- |
| Numeric | Numeric |
| Numeric | Categorical |
| Categorical | Categorical |
| Categorical | Numeric |

---

# STEP 1 — IDENTIFY TARGET TYPE

---

# Regression Problem

Target:

* Continuous numeric

Examples:

* Salary
* House price
* Revenue

---

# Classification Problem

Target:

* Categories/classes

Examples:

* Churn Yes/No
* Fraud/Not Fraud
* Disease category

---

# CASE A — NUMERICAL FEATURE vs NUMERICAL TARGET

Example:

* Income vs House Price

Goal:

> Does this feature affect the target?

---

# METHOD 1 — PEARSON CORRELATION

Measures:

* Linear relationship strength

Formula:

$$r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}$$

---

# Interpretation

| Correlation | Meaning |
| --- | --- |
| 0 | No linear relation |
| 0.2 | Weak |
| 0.5 | Moderate |
| 0.7+ | Strong |
| 1 | Perfect |

---

# Steps

```python
corr = df["Income"].corr(df["HousePrice"])

```

---

# Statistical Significance

Correlation alone is not enough.

Need:

* P-value

```python
from scipy.stats import pearsonr

corr, p = pearsonr(df["Income"], df["HousePrice"])

```

---

# Decision

```text
High |corr| + low p-value
= important feature

```

Usually:

```python
p < 0.05

```

means statistically significant.

---

# METHOD 2 — SPEARMAN CORRELATION

Use when:

* Nonlinear monotonic relationships
* Skewed data
* Outliers present

Instead of actual values:

* Uses ranks

```python
from scipy.stats import spearmanr

corr, p = spearmanr(x, y)

```

---

# WHEN TO USE PEARSON vs SPEARMAN

| Method | Use Case |
| --- | --- |
| Pearson | Linear + normal-ish |
| Spearman | Nonlinear/monotonic/skewed |

---

# CASE B — NUMERICAL FEATURE vs CATEGORICAL TARGET

VERY IMPORTANT.

Example:

* Salary vs Churn
* Age vs Disease

Goal:

> Are feature values different across classes?

---

# BINARY TARGET → T-TEST

Example:

* Churn = Yes/No

---

# Hypotheses

$H_0:$ Means are equal.

$H_1:$ Means differ.

---

# Formula

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

---

# Steps

```python
from scipy.stats import ttest_ind

group1 = df[df["Churn"] == 0]["Salary"]
group2 = df[df["Churn"] == 1]["Salary"]

t_stat, p = ttest_ind(group1, group2)

```

---

# Decision

```python
if p < 0.05:
    important feature

```

Because:

* Means differ significantly across classes.

---

# MULTI-CLASS TARGET → ANOVA

Example:

* Salary vs EducationLevel

Education:

* School
* College
* Masters
* PhD

---

# ANOVA Goal

Checks whether:

* At least one group mean differs

---

# Formula

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

---

# Steps

```python
from scipy.stats import f_oneway

f_stat, p = f_oneway(group1, group2, group3)

```

---

# Interpretation

| Result | Meaning |
| --- | --- |
| High F | Strong separation |
| Low p-value | Significant |

---

# CASE C — CATEGORICAL FEATURE vs CATEGORICAL TARGET

Example:

* Gender vs Churn

Goal:

> Are categories associated?

---

# METHOD — CHI-SQUARE TEST

Most important categorical test.

---

# Hypotheses

$H_0:$ No association.

$H_1:$ Association exists.

---

# Formula

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

---

# Steps

```python
from scipy.stats import chi2_contingency

table = pd.crosstab(df["Gender"], df["Churn"])

chi2, p, dof, expected = chi2_contingency(table)

```

---

# Interpretation

```python
p < 0.05

```

means:

* Significant association exists.

Feature matters.

---

# IMPORTANT PROBLEM WITH CHI-SQUARE

Chi-square significance depends on dataset size.

Huge datasets may make weak relationships appear significant.

So also calculate:

* Cramér’s V

---

# CRAMÉR’S V

Measures:

* Strength of categorical association

Range:

* 0 → no relation
* 1 → perfect relation

---

# Formula

$$V = \sqrt{\frac{\chi^2}{n(k-1)}}$$

---

# Interpretation

| Value | Meaning |
| --- | --- |
| <0.1 | Weak |
| 0.1–0.3 | Moderate |
| >0.5 | Strong |

---

# CASE D — CATEGORICAL FEATURE vs NUMERICAL TARGET

Example:

* City vs Revenue

---

# METHOD — ANOVA

Same idea:

* Compare target means across categories.

```python
f_oneway()

```

---

# CASE E — NONLINEAR RELATIONSHIPS

Correlation fails sometimes.

Example:

```text
y = x²

```

Pearson may show weak correlation.

But feature is clearly important.

---

# METHOD — MUTUAL INFORMATION

One of the BEST methods.

Measures:

* Information gain
* Nonlinear relationships

Works for:

* Numeric
* Categorical
* Mixed data

---

# Formula Concept

$MI(X,Y)$ Measures how much knowing X reduces uncertainty about Y.

---

# Steps

Classification:

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y)

```

Regression:

```python
from sklearn.feature_selection import mutual_info_regression

```

---

# Interpretation

| MI | Meaning |
| --- | --- |
| 0 | Independent |
| High | Important |

No upper limit.

Relative comparison matters.

---

# CASE F — TREE-BASED FEATURE IMPORTANCE

MOST PRACTICAL INDUSTRY METHOD.

---

# Random Forest Importance

Measures:

* Reduction in impurity

Features that split data well:

* Get higher importance.

---

# Steps

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X, y)

importance = model.feature_importances_

```

---

# Why Tree Importance is Powerful

Captures:

* Nonlinear relations
* Feature interactions
* Mixed data types

Very commonly used.

---

# PROBLEM WITH TREE IMPORTANCE

Bias toward:

* High-cardinality features
* Continuous variables

---

# BETTER VERSION → PERMUTATION IMPORTANCE

VERY reliable.

---

# IDEA

Shuffle one feature randomly.

If model performance drops heavily:

* Feature important.

If performance unchanged:

* Feature useless.

---

# Steps

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y)

```

---

# Why Permutation Importance is Excellent

It measures:

> Actual predictive contribution

Not statistical assumptions.

Very strong method.

---

# FEATURE IMPORTANCE MASTER FRAMEWORK

| Scenario | Statistical Method | Strength Measure |
| --- | --- | --- |
| Numeric vs Numeric | Pearson/Spearman | Correlation |
| Numeric vs Binary Category | T-test | P-value + Effect size |
| Numeric vs Multi-category | ANOVA | F-statistic |
| Categorical vs Categorical | Chi-square | Cramér’s V |
| Nonlinear relationships | Mutual Information | MI score |
| General ML importance | Random Forest | Feature importance |
| Best practical importance | Permutation Importance | Performance drop |

---

# INDUSTRY FEATURE IMPORTANCE FLOW

This is VERY close to real DS workflows.

```text
1. Identify feature type
2. Identify target type
3. Apply statistical significance test
4. Check strength metric
5. Apply nonlinear methods
6. Apply model-based importance
7. Validate using permutation importance
8. Remove weak/redundant features

```

---

# IMPORTANT CONCEPT — SIGNIFICANCE vs IMPORTANCE

A feature can be:

| Situation | Meaning |
| --- | --- |
| Significant but weak | Real effect, tiny impact |
| Important but nonlinear | Correlation misses it |
| Correlated but redundant | Duplicated information |

This is why:

* Multiple methods are used together.

---

# BEST PRACTICAL STACK FOR REAL EDA

Most strong data scientists use:

```text
1. Correlation/Spearman
2. Statistical tests
3. Mutual Information
4. Random Forest importance
5. Permutation Importance

```

Together.

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
