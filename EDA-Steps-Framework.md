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

## 0.1 — Quick Inspection

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

## 0.2 — Visualize Data Type Overview

Get a bird's-eye view of what you're working with.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart of data types
dtype_counts = df.dtypes.value_counts()
dtype_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Column Data Type Distribution')
plt.xlabel('Data Type')
plt.ylabel('Number of Columns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 0.3 — Visualize Missing Values Heatmap (First Look)

Before any treatment, see the full missing-value landscape in one shot.

```python
import missingno as msno

# Matrix view: white = missing
msno.matrix(df, figsize=(12, 6), color=(0.27, 0.52, 0.71))
plt.title('Missing Value Matrix (White = Missing)')
plt.tight_layout()
plt.show()

# Bar chart view: shows % present per column
msno.bar(df, figsize=(12, 5))
plt.title('Data Completeness per Column')
plt.tight_layout()
plt.show()
```

---

# STEP 1 — HANDLE MISSING VALUES

This is ALWAYS one of the first steps.

---

# PART A — NUMERICAL MISSING VALUES FRAMEWORK

---

## Step 1 — Find Missing Percentage

```python
missing_percent = df[col].isnull().mean() * 100
```

### 1A.1 — Visualize Missing % Across All Columns

```python
# Horizontal bar chart of missing % per column
missing_df = df.isnull().mean() * 100
missing_df = missing_df[missing_df > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing_df.plot(kind='barh', color='salmon', edgecolor='black')
plt.axvline(x=5,  color='green',  linestyle='--', label='5% threshold')
plt.axvline(x=30, color='orange', linestyle='--', label='30% threshold')
plt.axvline(x=60, color='red',    linestyle='--', label='60% threshold')
plt.title('Missing Value % per Column')
plt.xlabel('Missing %')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Step 2 — Analyze Distribution

Check:

- Histogram
- Boxplot
- Skewness

```python
sns.histplot(df[col])
sns.boxplot(x=df[col])
df[col].skew()
```

### 1A.2 — Visualize Distribution + Skewness

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='steelblue')
axes[0].axvline(df[col].mean(),   color='red',    linestyle='--', label=f'Mean: {df[col].mean():.2f}')
axes[0].axvline(df[col].median(), color='green',  linestyle='--', label=f'Median: {df[col].median():.2f}')
axes[0].set_title(f'{col} — Distribution (Skew: {df[col].skew():.2f})')
axes[0].legend()

# Boxplot
sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightblue')
axes[1].set_title(f'{col} — Boxplot')

plt.tight_layout()
plt.show()
```

---

## Step 3 — Decide Missing Value Strategy

---

# CASE 1 — Missing Values SMALL (< 5%)

Usually safe to impute.

---

### If distribution is NORMAL

Conditions:

```python
abs(skewness) < 0.5
```

Use:

- Mean imputation

```python
df[col].fillna(df[col].mean())
```

Why?

- Mean represents center well in symmetric data.

#### Visualization — Before vs After Mean Imputation

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before imputation
sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='salmon')
axes[0].axvline(df[col].mean(), color='blue', linestyle='--', label='Mean')
axes[0].set_title(f'{col} — Before Imputation')
axes[0].legend()

# After imputation
df_imputed = df[col].fillna(df[col].mean())
sns.histplot(df_imputed, kde=True, ax=axes[1], color='steelblue')
axes[1].axvline(df_imputed.mean(), color='blue', linestyle='--', label='Mean')
axes[1].set_title(f'{col} — After Mean Imputation')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

### If distribution is SKEWED

Conditions:

```python
abs(skewness) > 0.5
```

Use:

- Median imputation

```python
df[col].fillna(df[col].median())
```

Why?

- Median resistant to outliers.

#### Visualization — Skewed Distribution + Median Imputation

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before imputation — annotate skewness
sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='salmon')
axes[0].axvline(df[col].mean(),   color='blue',  linestyle='--', label=f'Mean: {df[col].mean():.2f}')
axes[0].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
axes[0].set_title(f'{col} — Skewed (skew={df[col].skew():.2f})')
axes[0].legend()

# After imputation
df_imputed = df[col].fillna(df[col].median())
sns.histplot(df_imputed, kde=True, ax=axes[1], color='steelblue')
axes[1].axvline(df_imputed.median(), color='green', linestyle='--', label='Median')
axes[1].set_title(f'{col} — After Median Imputation')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

# CASE 2 — Missing Values MODERATE (5%–30%)

Now think carefully.

---

### Option A — Median/Mean Imputation

Still okay if:

- Feature important
- Missingness random

---

### Option B — Predictive Imputation

Use ML models:

- KNN Imputer
- Iterative Imputer

Used when:

- Column highly important
- Strong feature relationships exist

#### Visualization — Missingness Pattern Before Predictive Imputation

```python
import missingno as msno

# Correlation heatmap of missingness: which columns miss together?
msno.heatmap(df, figsize=(10, 8), cmap='coolwarm')
plt.title('Missingness Correlation Between Columns')
plt.tight_layout()
plt.show()
```

---

### Option C — Add Missing Indicator

```python
df[col+"_missing"] = df[col].isnull().astype(int)
```

Sometimes missing itself carries information.

Example:

- Missing income may indicate unemployed customers.

#### Visualization — Missing Indicator vs Target

```python
# Does missingness correlate with target?
df['missing_flag'] = df[col].isnull().astype(int)

plt.figure(figsize=(8, 5))
sns.barplot(x='missing_flag', y=target_col, data=df, palette='Set2')
plt.title(f'Target Mean by {col} Missingness\n(0 = Present, 1 = Missing)')
plt.xlabel('Missing Flag')
plt.ylabel(f'Mean {target_col}')
plt.tight_layout()
plt.show()
```

---

# CASE 3 — Missing Values VERY HIGH (> 40–60%)

Ask:

- Is feature business critical?

If NO:

- Drop column

If YES:

- Advanced imputation
- Domain knowledge

#### Visualization — Assess Before Dropping

```python
# Visualize what data remains in high-missing columns
high_missing_cols = [c for c in df.columns if df[c].isnull().mean() > 0.4]

for col in high_missing_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Distribution of available data
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{col} — Available Data ({df[col].notna().sum()} rows)')

    # Missing vs present in relation to target
    df['_temp_flag'] = df[col].isnull().map({True: 'Missing', False: 'Present'})
    sns.countplot(x='_temp_flag', hue=target_col, data=df, ax=axes[1], palette='Set2')
    axes[1].set_title(f'{col} — Missing/Present by Target')
    df.drop(columns=['_temp_flag'], inplace=True)

    plt.tight_layout()
    plt.show()
```

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

## Step 1 — Count Missing %

```python
df[col].isnull().mean()
```

---

## Step 2 — Analyze Cardinality

Check:

```python
df[col].nunique()
```

### 1B.2 — Visualize Cardinality Across All Categorical Columns

```python
cat_cardinality = df[categorical_cols].nunique().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
cat_cardinality.plot(kind='bar', color='mediumpurple', edgecolor='black')
plt.title('Cardinality (Unique Values) per Categorical Column')
plt.xlabel('Column')
plt.ylabel('Unique Values Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

## Step 3 — Decide Strategy

---

# CASE 1 — Small Missing %

Use:

- Mode imputation

```python
df[col].fillna(df[col].mode()[0])
```

Best for:

- Low missing %
- Stable categories

#### Visualization — Category Frequency Before vs After Mode Imputation

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before imputation
df[col].value_counts().plot(kind='bar', ax=axes[0], color='salmon', edgecolor='black')
axes[0].set_title(f'{col} — Before Mode Imputation')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')

# After imputation
df[col].fillna(df[col].mode()[0]).value_counts().plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title(f'{col} — After Mode Imputation')
axes[1].set_xlabel('Category')

plt.tight_layout()
plt.show()
```

---

# CASE 2 — Missing May Have Meaning

Create separate category:

```python
df[col].fillna("Unknown")
```

Example:

- Occupation missing
- Education missing

Very common in industry.

#### Visualization — "Unknown" Category vs Target

```python
df_temp = df.copy()
df_temp[col] = df_temp[col].fillna("Unknown")

plt.figure(figsize=(10, 5))
sns.countplot(x=col, hue=target_col, data=df_temp, palette='Set2',
              order=df_temp[col].value_counts().index)
plt.title(f'{col} (with Unknown) vs {target_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

# CASE 3 — High Missing %

If:

- Too sparse
- Too many unique values
- Low business value

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

## Step 1 — Visualize Distribution

```python
sns.histplot(df[col], kde=True)
sns.boxplot(x=df[col])
```

### 2A.1 — Full Distribution Dashboard per Column

```python
for col in numerical_cols:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Histogram + KDE
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{col} — Histogram')

    # Boxplot
    sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightblue')
    axes[1].set_title(f'{col} — Boxplot')

    # QQ-Plot (normality check)
    from scipy import stats
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title(f'{col} — Q-Q Plot')

    plt.suptitle(f'Distribution Analysis: {col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

---

## Step 2 — Check Skewness

```python
df[col].skew()
```

| Skewness | Meaning |
| --- | --- |
| 0 | Symmetric |
| > 0 | Right skew |
| < 0 | Left skew |

| Value | Interpretation |
| --- | --- |
| -0.5 to 0.5 | Approximately normal |
| 0.5 to 1 | Moderately skewed |
| >1 | Highly skewed |

### 2A.2 — Visualize Skewness Across All Numerical Columns

```python
skewness = df[numerical_cols].skew().sort_values()

colors = ['green' if abs(s) < 0.5 else ('orange' if abs(s) < 1 else 'red')
          for s in skewness]

plt.figure(figsize=(10, 6))
skewness.plot(kind='barh', color=colors, edgecolor='black')
plt.axvline(x=0.5,  color='orange', linestyle='--', label='Moderate skew boundary')
plt.axvline(x=-0.5, color='orange', linestyle='--')
plt.axvline(x=1.0,  color='red',    linestyle='--', label='High skew boundary')
plt.axvline(x=-1.0, color='red',    linestyle='--')
plt.title('Skewness per Numerical Column\n(Green=Normal, Orange=Moderate, Red=Highly Skewed)')
plt.xlabel('Skewness')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Step 3 — Check Kurtosis

```python
df[col].kurtosis()
```

Measures:

- Tail heaviness

| Kurtosis | Meaning |
| --- | --- |
| ~0 | Normal |
| High | Heavy tails/outliers |
| Low | Flat distribution |

### 2A.3 — Visualize Kurtosis Across All Numerical Columns

```python
kurtosis_vals = df[numerical_cols].kurtosis().sort_values()

plt.figure(figsize=(10, 6))
kurtosis_vals.plot(kind='barh', color='mediumpurple', edgecolor='black')
plt.axvline(x=3, color='blue', linestyle='--', label='Normal kurtosis (=3)')
plt.title('Kurtosis per Numerical Column\n(High kurtosis = heavy tails / more outliers)')
plt.xlabel('Kurtosis')
plt.legend()
plt.tight_layout()
plt.show()
```

---

# PART B — CHOOSE OUTLIER METHOD

---

# CASE 1 — Approximately Normal Distribution

Conditions:

```python
abs(skewness) < 0.5
```

Use:

- Z-score

---

## Z-Score Method

$$z = \frac{x-\mu}{\sigma}$$

```python
from scipy.stats import zscore

z = np.abs(zscore(df[col]))

outliers = df[z > 3]
```

Use when:

- Bell-shaped distribution
- Gaussian-like data

### 2B.1 — Visualize Z-Score Outliers

```python
from scipy.stats import zscore
import numpy as np

z_scores = np.abs(zscore(df[col].dropna()))
threshold = 3

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot highlighting outliers
idx = df[col].dropna().index
colors = ['red' if z > threshold else 'steelblue' for z in z_scores]
axes[0].scatter(range(len(df[col].dropna())), df[col].dropna(), c=colors, alpha=0.6, s=15)
axes[0].set_title(f'{col} — Z-Score Outliers (Red = |z| > {threshold})')
axes[0].set_xlabel('Index')
axes[0].set_ylabel(col)

# Z-score distribution
axes[1].hist(z_scores, bins=50, color='steelblue', edgecolor='black')
axes[1].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
axes[1].set_title(f'{col} — Z-Score Distribution')
axes[1].set_xlabel('|Z-Score|')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Outliers detected (|z| > {threshold}): {(z_scores > threshold).sum()}")
```

---

# CASE 2 — Skewed Distribution

Conditions:

```python
abs(skewness) > 0.5
```

Use:

- IQR

---

## IQR Method

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

### 2B.2 — Visualize IQR Outlier Boundaries

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot with IQR annotation
sns.boxplot(x=df[col].dropna(), ax=axes[0], color='lightblue',
            flierprops=dict(marker='o', color='red', alpha=0.5))
axes[0].axvline(lower, color='red',    linestyle='--', label=f'Lower: {lower:.2f}')
axes[0].axvline(upper, color='orange', linestyle='--', label=f'Upper: {upper:.2f}')
axes[0].set_title(f'{col} — IQR Boundaries')
axes[0].legend()

# Histogram with outlier zones shaded
sns.histplot(df[col].dropna(), kde=True, ax=axes[1], color='steelblue')
axes[1].axvspan(df[col].min(), lower, alpha=0.2, color='red',    label='Outlier zone (low)')
axes[1].axvspan(upper, df[col].max(), alpha=0.2, color='orange', label='Outlier zone (high)')
axes[1].set_title(f'{col} — Outlier Zones')
axes[1].legend()

plt.tight_layout()
plt.show()

outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
print(f"IQR Outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
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

- Data clearly invalid

Example:

- Age = 500

---

# Option 2 — Capping (Winsorization)

Replace extremes.

```python
df[col] = np.clip(df[col], lower, upper)
```

Most common industry solution.

### 2C.2 — Visualize Before vs After Capping

```python
df_capped = df[col].clip(lower, upper)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='salmon')
axes[0].set_title(f'{col} — Before Capping')
axes[0].axvline(lower, color='red',    linestyle='--', label='Lower bound')
axes[0].axvline(upper, color='orange', linestyle='--', label='Upper bound')
axes[0].legend()

sns.histplot(df_capped, kde=True, ax=axes[1], color='steelblue')
axes[1].set_title(f'{col} — After Winsorization')

plt.tight_layout()
plt.show()
```

---

# Option 3 — Log Transform

For right-skewed data.

```python
df[col] = np.log1p(df[col])
```

Reduces skewness.

### 2C.3 — Visualize Before vs After Log Transform

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='salmon')
axes[0].set_title(f'{col} — Before Log Transform (skew={df[col].skew():.2f})')

log_col = np.log1p(df[col].dropna())
sns.histplot(log_col, kde=True, ax=axes[1], color='steelblue')
axes[1].set_title(f'log1p({col}) — After Transform (skew={log_col.skew():.2f})')

plt.tight_layout()
plt.show()
```

---

# Option 4 — Keep Outliers

Sometimes outliers are valuable.

Example:

- Fraud detection
- Financial spikes

Never blindly remove outliers.

### 2C.4 — Visualize Outliers vs Target (Before Deciding to Remove)

```python
# Are outliers in a specific class? (for classification problems)
outlier_mask = (df[col] < lower) | (df[col] > upper)

plt.figure(figsize=(8, 5))
sns.countplot(x=target_col, hue=outlier_mask.map({True: 'Outlier', False: 'Normal'}),
              data=df, palette={'Outlier': 'red', 'Normal': 'steelblue'})
plt.title(f'Outlier Distribution Across {target_col} Classes\n(Decide carefully before removing!)')
plt.tight_layout()
plt.show()
```

---

# CATEGORICAL "OUTLIERS"

Not true outliers, but:

- Rare categories
- High cardinality issues

---

## Detect Rare Categories

```python
df[col].value_counts(normalize=True)
```

### Visualization — Rare Category Detection

```python
# Visualize category frequency distribution
freq = df[col].value_counts(normalize=True) * 100

plt.figure(figsize=(12, 5))
bars = freq.plot(kind='bar', color='steelblue', edgecolor='black')
plt.axhline(y=1.0, color='red', linestyle='--', label='Rare threshold (1%)')
plt.title(f'{col} — Category Frequency Distribution')
plt.xlabel('Category')
plt.ylabel('Frequency (%)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Print rare categories
rare = freq[freq < 1.0]
print(f"\nRare categories (<1%):\n{rare}")
```

Rare categories may be grouped:

```python
rare_cats = freq[freq < 0.01].index
df[col] = df[col].replace(rare_cats, 'Other')
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

---

## 3.0 — Overall Feature Correlation Heatmap (Start Here)

Always begin with a macro view of relationships.

```python
plt.figure(figsize=(14, 10))
corr_matrix = df[numerical_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title('Pairwise Correlation Heatmap (Numerical Features)')
plt.tight_layout()
plt.show()
```

---

## 3.1 — Pairplot (Visual Relationship Overview)

Ideal for datasets with < 10 numerical features.

```python
sns.pairplot(df[numerical_cols + [target_col]],
             hue=target_col if df[target_col].nunique() <= 5 else None,
             diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot — Feature Relationships', y=1.02)
plt.tight_layout()
plt.show()
```

---

# STEP 1 — IDENTIFY TARGET TYPE

---

## Regression Problem

Target:

- Continuous numeric

Examples:

- Salary
- House price
- Revenue

---

## Classification Problem

Target:

- Categories/classes

Examples:

- Churn Yes/No
- Fraud/Not Fraud
- Disease category

### 3.1 — Visualize Target Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Numeric target: histogram
if df[target_col].nunique() > 10:
    sns.histplot(df[target_col], kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{target_col} — Distribution (Regression Target)')
    axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', label='Mean')
    axes[0].legend()

    # QQ plot
    from scipy import stats
    stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f'{target_col} — Q-Q Plot')

# Categorical target: countplot
else:
    sns.countplot(x=target_col, data=df, ax=axes[0], palette='Set2')
    axes[0].set_title(f'{target_col} — Class Distribution')

    # Pie chart
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                        ax=axes[1], colors=sns.color_palette('Set2'))
    axes[1].set_title(f'{target_col} — Class Proportions')
    axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

# CASE A — NUMERICAL FEATURE vs NUMERICAL TARGET

Example:

- Income vs House Price

Goal:

> Does this feature affect the target?

---

## METHOD 1 — PEARSON CORRELATION

Measures:

- Linear relationship strength

$$r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}$$

| Correlation | Meaning |
| --- | --- |
| 0 | No linear relation |
| 0.2 | Weak |
| 0.5 | Moderate |
| 0.7+ | Strong |
| 1 | Perfect |

```python
corr = df["Income"].corr(df["HousePrice"])
```

Statistical Significance:

```python
from scipy.stats import pearsonr

corr, p = pearsonr(df["Income"], df["HousePrice"])
```

### Visualization — Scatter Plot with Regression Line

```python
from scipy.stats import pearsonr

corr, p = pearsonr(df[feature_col].dropna(), df[target_col].dropna())

plt.figure(figsize=(8, 6))
sns.regplot(x=feature_col, y=target_col, data=df,
            scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title(f'{feature_col} vs {target_col}\nPearson r = {corr:.3f}, p = {p:.4f}')
plt.tight_layout()
plt.show()
```

### Visualization — Correlation Heatmap with Target Column

```python
# Correlations of ALL features with target, sorted
target_corr = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
target_corr.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title(f'Pearson |Correlation| with {target_col}')
plt.xlabel('|Correlation|')
plt.axvline(x=0.5, color='green',  linestyle='--', label='Moderate (0.5)')
plt.axvline(x=0.7, color='orange', linestyle='--', label='Strong (0.7)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## METHOD 2 — SPEARMAN CORRELATION

Use when:

- Nonlinear monotonic relationships
- Skewed data
- Outliers present

```python
from scipy.stats import spearmanr

corr, p = spearmanr(x, y)
```

| Method | Use Case |
| --- | --- |
| Pearson | Linear + normal-ish |
| Spearman | Nonlinear/monotonic/skewed |

### Visualization — Pearson vs Spearman Comparison

```python
from scipy.stats import pearsonr, spearmanr

pearson_r,  p_pear  = pearsonr(df[feature_col].dropna(), df[target_col].dropna())
spearman_r, p_spear = spearmanr(df[feature_col].dropna(), df[target_col].dropna())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pearson: raw values
sns.regplot(x=feature_col, y=target_col, data=df, ax=axes[0],
            scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
axes[0].set_title(f'Pearson r = {pearson_r:.3f} (p={p_pear:.4f})\n[Linear relationship]')

# Spearman: rank values
df_ranked = df[[feature_col, target_col]].dropna().rank()
sns.regplot(x=feature_col, y=target_col, data=df_ranked, ax=axes[1],
            scatter_kws={'alpha': 0.4, 'color': 'orange'}, line_kws={'color': 'red'})
axes[1].set_title(f'Spearman r = {spearman_r:.3f} (p={p_spear:.4f})\n[Monotonic relationship via ranks]')

plt.tight_layout()
plt.show()
```

---

# CASE B — NUMERICAL FEATURE vs CATEGORICAL TARGET

VERY IMPORTANT.

Example:

- Salary vs Churn
- Age vs Disease

Goal:

> Are feature values different across classes?

---

## BINARY TARGET → T-TEST

Example:

- Churn = Yes/No

**Hypotheses:**
$H_0:$ Means are equal.  
$H_1:$ Means differ.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

```python
from scipy.stats import ttest_ind

group1 = df[df["Churn"] == 0]["Salary"]
group2 = df[df["Churn"] == 1]["Salary"]

t_stat, p = ttest_ind(group1, group2)
```

### Visualization — Distribution per Class + T-Test Result

```python
from scipy.stats import ttest_ind

group0 = df[df[target_col] == 0][feature_col].dropna()
group1 = df[df[target_col] == 1][feature_col].dropna()
t_stat, p = ttest_ind(group0, group1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE overlay per class
sns.kdeplot(group0, ax=axes[0], label='Class 0', fill=True, alpha=0.4, color='steelblue')
sns.kdeplot(group1, ax=axes[0], label='Class 1', fill=True, alpha=0.4, color='salmon')
axes[0].axvline(group0.mean(), color='blue',   linestyle='--', label=f'Mean(0)={group0.mean():.1f}')
axes[0].axvline(group1.mean(), color='red',    linestyle='--', label=f'Mean(1)={group1.mean():.1f}')
axes[0].set_title(f'{feature_col} by {target_col}\nt={t_stat:.3f}, p={p:.4f}')
axes[0].legend()

# Boxplot per class
sns.boxplot(x=target_col, y=feature_col, data=df, ax=axes[1], palette='Set2')
axes[1].set_title(f'{feature_col} Boxplot by {target_col} Class')

plt.tight_layout()
plt.show()
```

---

## MULTI-CLASS TARGET → ANOVA

Example:

- Salary vs EducationLevel (School / College / Masters / PhD)

ANOVA checks whether at least one group mean differs.

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

```python
from scipy.stats import f_oneway

f_stat, p = f_oneway(group1, group2, group3)
```

| Result | Meaning |
| --- | --- |
| High F | Strong separation |
| Low p-value | Significant |

### Visualization — ANOVA Boxplot + Violin Plot

```python
from scipy.stats import f_oneway

groups = [df[df[target_col] == cat][feature_col].dropna()
          for cat in df[target_col].unique()]
f_stat, p = f_oneway(*groups)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot per class
sns.boxplot(x=target_col, y=feature_col, data=df, ax=axes[0], palette='Set2')
axes[0].set_title(f'{feature_col} by {target_col}\nANOVA F={f_stat:.3f}, p={p:.4f}')
axes[0].tick_params(axis='x', rotation=30)

# Violin plot (richer shape)
sns.violinplot(x=target_col, y=feature_col, data=df, ax=axes[1],
               palette='Set2', inner='quartile')
axes[1].set_title(f'{feature_col} by {target_col} — Violin Plot')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
```

---

# CASE C — CATEGORICAL FEATURE vs CATEGORICAL TARGET

Example:

- Gender vs Churn

Goal:

> Are categories associated?

---

## METHOD — CHI-SQUARE TEST

$H_0:$ No association.  
$H_1:$ Association exists.

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

```python
from scipy.stats import chi2_contingency

table = pd.crosstab(df["Gender"], df["Churn"])

chi2, p, dof, expected = chi2_contingency(table)
```

p < 0.05 means significant association exists.

### Visualization — Contingency Heatmap + Stacked Bar

```python
from scipy.stats import chi2_contingency

ct = pd.crosstab(df[feature_col], df[target_col])
chi2, p, dof, expected = chi2_contingency(ct)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap of observed counts
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[0], linewidths=0.5)
axes[0].set_title(f'Contingency Table: {feature_col} vs {target_col}\nχ²={chi2:.2f}, p={p:.4f}')

# Normalized stacked bar (proportions per category)
ct_norm = ct.div(ct.sum(axis=1), axis=0)
ct_norm.plot(kind='bar', stacked=True, ax=axes[1],
             colormap='Set2', edgecolor='black')
axes[1].set_title(f'{feature_col} vs {target_col} — Proportions per Category')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title=target_col)

plt.tight_layout()
plt.show()
```

---

## IMPORTANT PROBLEM WITH CHI-SQUARE

Chi-square significance depends on dataset size. Huge datasets may make weak relationships appear significant.

So also calculate: **Cramér's V**

---

## CRAMÉR'S V

$$V = \sqrt{\frac{\chi^2}{n(k-1)}}$$

| Value | Meaning |
| --- | --- |
| <0.1 | Weak |
| 0.1–0.3 | Moderate |
| >0.5 | Strong |

### Visualization — Cramér's V Across All Categorical Features

```python
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

cramer_scores = {col: cramers_v(df[col], df[target_col])
                 for col in categorical_cols}
cramer_series = pd.Series(cramer_scores).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
colors = ['green' if v > 0.5 else ('orange' if v > 0.1 else 'salmon')
          for v in cramer_series]
cramer_series.plot(kind='barh', color=colors, edgecolor='black')
plt.axvline(x=0.1, color='orange', linestyle='--', label='Moderate (0.1)')
plt.axvline(x=0.5, color='green',  linestyle='--', label='Strong (0.5)')
plt.title(f"Cramér's V — Categorical Feature Association with {target_col}")
plt.xlabel("Cramér's V")
plt.legend()
plt.tight_layout()
plt.show()
```

---

# CASE D — CATEGORICAL FEATURE vs NUMERICAL TARGET

Example:

- City vs Revenue

---

## METHOD — ANOVA

Same idea: compare target means across categories.

```python
f_oneway()
```

### Visualization — Category vs Numeric Target with Point Estimates

```python
from scipy.stats import f_oneway

groups = [df[df[feature_col] == cat][target_col].dropna()
          for cat in df[feature_col].unique()]
f_stat, p = f_oneway(*groups)

plt.figure(figsize=(12, 5))
order = df.groupby(feature_col)[target_col].median().sort_values(ascending=False).index

sns.boxplot(x=feature_col, y=target_col, data=df, order=order, palette='Set3')
sns.stripplot(x=feature_col, y=target_col, data=df, order=order,
              color='black', alpha=0.3, size=3, jitter=True)
plt.title(f'{feature_col} vs {target_col}\nANOVA F={f_stat:.3f}, p={p:.4f}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

# CASE E — NONLINEAR RELATIONSHIPS

Correlation fails sometimes.

Example:

```text
y = x²
```

Pearson may show weak correlation. But feature is clearly important.

---

## METHOD — MUTUAL INFORMATION

One of the BEST methods.

$MI(X,Y)$ — Measures how much knowing X reduces uncertainty about Y.

Classification:

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y)
```

Regression:

```python
from sklearn.feature_selection import mutual_info_regression
```

| MI | Meaning |
| --- | --- |
| 0 | Independent |
| High | Important |

### Visualization — Mutual Information Bar Chart

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

X = df[numerical_cols].fillna(df[numerical_cols].median())
y = df[target_col]

# Use appropriate function based on problem type
if df[target_col].nunique() <= 10:   # classification
    mi_scores = mutual_info_classif(X, y, random_state=42)
else:                                 # regression
    mi_scores = mutual_info_regression(X, y, random_state=42)

mi_series = pd.Series(mi_scores, index=numerical_cols).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
mi_series.plot(kind='barh', color='teal', edgecolor='black')
plt.title(f'Mutual Information Scores with {target_col}\n(Captures linear AND nonlinear relationships)')
plt.xlabel('Mutual Information Score')
plt.tight_layout()
plt.show()
```

### Visualization — Side-by-Side: Correlation vs Mutual Information

```python
# Compare: features that Pearson misses but MI catches (nonlinear)
pearson_scores = df[numerical_cols].corrwith(df[target_col]).abs()

comparison = pd.DataFrame({
    'Pearson |r|': pearson_scores,
    'Mutual Info': mi_series
}).sort_values('Mutual Info', ascending=False)

comparison.plot(kind='barh', figsize=(12, 7), colormap='Set1', edgecolor='black')
plt.title('Feature Importance: Pearson vs Mutual Information\n(Divergence reveals nonlinear relationships)')
plt.xlabel('Score')
plt.tight_layout()
plt.show()
```

---

# CASE F — TREE-BASED FEATURE IMPORTANCE

MOST PRACTICAL INDUSTRY METHOD.

---

## Random Forest Importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importance = model.feature_importances_
```

### Visualization — Random Forest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

X = df[numerical_cols].fillna(df[numerical_cols].median())
y = df[target_col]

if df[target_col].nunique() <= 10:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X, y)

feat_imp = pd.Series(model.feature_importances_, index=numerical_cols).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='darkorange', edgecolor='black')
plt.title(f'Random Forest Feature Importances\n(Based on impurity reduction)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

---

## BETTER VERSION → PERMUTATION IMPORTANCE

VERY reliable.

**Idea:** Shuffle one feature. If model performance drops: feature is important. If unchanged: feature is useless.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y)
```

### Visualization — Permutation Importance with Confidence Intervals

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

perm_imp = pd.DataFrame({
    'mean': result.importances_mean,
    'std':  result.importances_std
}, index=numerical_cols).sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(perm_imp.index, perm_imp['mean'],
        xerr=perm_imp['std'], color='steelblue',
        edgecolor='black', capsize=4)
ax.axvline(x=0, color='red', linestyle='--', label='No impact baseline')
ax.set_title(f'Permutation Importance with Std Dev\n(More reliable than impurity-based)')
ax.set_xlabel('Mean Accuracy Drop When Feature Shuffled')
ax.legend()
plt.tight_layout()
plt.show()
```

### Visualization — RF Importance vs Permutation Importance Side-by-Side

```python
# Spot features that look important in RF but aren't truly predictive
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

feat_imp.plot(kind='barh', ax=axes[0], color='darkorange', edgecolor='black')
axes[0].set_title('Random Forest Feature Importance\n(May be biased toward high-cardinality)')
axes[0].set_xlabel('Importance Score')

perm_imp['mean'].sort_values().plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Permutation Importance\n(Actual predictive contribution)')
axes[1].set_xlabel('Mean Performance Drop')

plt.suptitle('RF Importance vs Permutation Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

# FEATURE IMPORTANCE MASTER FRAMEWORK

| Scenario | Statistical Method | Strength Measure |
| --- | --- | --- |
| Numeric vs Numeric | Pearson/Spearman | Correlation |
| Numeric vs Binary Category | T-test | P-value + Effect size |
| Numeric vs Multi-category | ANOVA | F-statistic |
| Categorical vs Categorical | Chi-square | Cramér's V |
| Nonlinear relationships | Mutual Information | MI score |
| General ML importance | Random Forest | Feature importance |
| Best practical importance | Permutation Importance | Performance drop |

---

# INDUSTRY FEATURE IMPORTANCE FLOW

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

## Final Step — Combined Feature Importance Dashboard

```python
# Bring everything together in one ranked summary table
summary = pd.DataFrame({
    'Pearson |r|':     pearson_scores,
    'Spearman |r|':    df[numerical_cols].corrwith(df[target_col], method='spearman').abs(),
    'Mutual Info':     mi_series,
    'RF Importance':   feat_imp,
    'Perm Importance': perm_imp['mean']
})

# Rank each method (1 = most important)
ranked = summary.rank(ascending=False).mean(axis=1).sort_values()

plt.figure(figsize=(10, 6))
ranked.plot(kind='barh', color='mediumseagreen', edgecolor='black')
plt.title('Average Rank Across All Importance Methods\n(Lower = More Important)')
plt.xlabel('Average Rank (lower is better)')
plt.tight_layout()
plt.show()

print("\n--- Combined Feature Importance Summary ---")
print(summary.round(4).to_string())
```

---

# IMPORTANT CONCEPT — SIGNIFICANCE vs IMPORTANCE

A feature can be:

| Situation | Meaning |
| --- | --- |
| Significant but weak | Real effect, tiny impact |
| Important but nonlinear | Correlation misses it |
| Correlated but redundant | Duplicated information |

This is why multiple methods are used together.

---

## Redundancy Check — Visualize Highly Correlated Features

```python
# Flag pairs with |correlation| > threshold for potential removal
corr_matrix = df[numerical_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                   for col in upper_tri.columns
                   for row in upper_tri.index
                   if upper_tri.loc[row, col] > 0.85]

print("Highly correlated pairs (|r| > 0.85):")
for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: -x[2]):
    print(f"  {feat1} — {feat2}: {corr_val:.3f}")

# Cluster heatmap to visually spot redundant groups
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.clustermap(df[numerical_cols].corr(), cmap='coolwarm', center=0,
               annot=True, fmt='.2f', figsize=(12, 10))
plt.suptitle('Clustered Correlation Map — Spot Redundant Feature Groups', y=1.02)
plt.show()
```

---

# GOLDEN RULES OF EDA

---

**Rule 1** — Never blindly apply techniques. Always understand distribution first.

**Rule 2** — Visualization before statistics. Humans detect patterns visually faster.

**Rule 3** — Business understanding matters more than formulas.

**Rule 4** — Outliers are not always bad.

**Rule 5** — Correlation does NOT imply causation.

**Rule 6** — Missing data itself may contain information.

---

# FINAL UNIVERSAL EDA CHECKLIST

```text
1.  Understand business problem
2.  Identify target variable
3.  Identify numerical/categorical columns
4.  Analyze missing values
5.  Handle missing values
6.  Analyze distributions
7.  Detect skewness/kurtosis
8.  Detect outliers
9.  Handle outliers
10. Analyze feature-target relationships
11. Statistical significance testing
12. Correlation analysis
13. Feature importance analysis
14. Remove redundancy
15. Feature engineering
16. Prepare for ML
```
