I'll create a comprehensive EDA guide markdown file for you. Let me write this out as a senior data scientist would for a junior team member. 

Read memory

I'm in Ask mode and can't create files, but I can provide the full markdown content here for you to save. Here's the complete EDA guide:

---

# 🔬 Day-to-Day EDA Guide for Beginners

**Author:** Senior Data Scientist  
**Audience:** Junior Data Scientists  
**Philosophy:** "Understand your data before you model it."

---

## Setup — Common Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_columns', None)

# Load your data
df = pd.read_csv("your_data.csv")
```

---

## 1. Numerical Feature vs Numerical Target

> Example: `age` (feature) → `salary` (target)

### 1.1 Handling Missing Data

**Step 1: Detect missing values**

```python
feature_col = "age"
target_col = "salary"

# Check missing counts
print(df[[feature_col, target_col]].isnull().sum())
print(f"\nMissing % in feature: {df[feature_col].isnull().mean()*100:.2f}%")
print(f"Missing % in target: {df[target_col].isnull().mean()*100:.2f}%")
```

**Step 2: Visualize missingness pattern**

```python
import missingno as msno
msno.matrix(df[[feature_col, target_col]])
plt.title("Missing Data Pattern")
plt.show()
```

**Step 3: Determine missingness type (MCAR, MAR, MNAR)**

```python
# Little's MCAR test approximation
# If missing values in feature are independent of target values:
missing_mask = df[feature_col].isnull()
if df[target_col].notna().sum() > 0:
    group_with_missing = df.loc[missing_mask, target_col].dropna()
    group_without_missing = df.loc[~missing_mask, target_col].dropna()
    
    if len(group_with_missing) > 0 and len(group_without_missing) > 0:
        stat, p_value = stats.ttest_ind(group_with_missing, group_without_missing)
        print(f"T-test p-value: {p_value:.4f}")
        print("MCAR likely" if p_value > 0.05 else "MAR/MNAR likely — missingness related to target")
```

**Step 4: Imputation strategies**

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Strategy 1: Mean/Median (use median if skewed)
skewness = df[feature_col].skew()
print(f"Skewness: {skewness:.2f}")

if abs(skewness) < 0.5:
    imputer = SimpleImputer(strategy='mean')
else:
    imputer = SimpleImputer(strategy='median')

df[feature_col + '_imputed'] = imputer.fit_transform(df[[feature_col]])

# Strategy 2: KNN Imputer (better when features are correlated)
knn_imputer = KNNImputer(n_neighbors=5)
df[feature_col + '_knn'] = knn_imputer.fit_transform(df[[feature_col, target_col]])[:, 0]

# Strategy 3: Add missing indicator flag
df[feature_col + '_was_missing'] = df[feature_col].isnull().astype(int)
```

**Visualization: Before vs After imputation**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df[feature_col].dropna(), bins=30, alpha=0.7, label='Original')
axes[0].set_title("Before Imputation")
axes[0].legend()

axes[1].hist(df[feature_col + '_imputed'], bins=30, alpha=0.7, color='orange', label='Imputed')
axes[1].set_title("After Imputation")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

### 1.2 Detecting and Handling Outliers

**Step 1: Visual detection**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot
sns.boxplot(x=df[feature_col], ax=axes[0])
axes[0].set_title(f"Boxplot of {feature_col}")

# Histogram with KDE
sns.histplot(df[feature_col], kde=True, ax=axes[1])
axes[1].set_title(f"Distribution of {feature_col}")

# Scatter plot against target
axes[2].scatter(df[feature_col], df[target_col], alpha=0.5)
axes[2].set_xlabel(feature_col)
axes[2].set_ylabel(target_col)
axes[2].set_title(f"{feature_col} vs {target_col}")

plt.tight_layout()
plt.show()
```

**Step 2: Statistical detection methods**

```python
# Method 1: IQR Method
Q1 = df[feature_col].quantile(0.25)
Q3 = df[feature_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df[feature_col] < lower_bound) | (df[feature_col] > upper_bound)]
print(f"IQR Method: {len(outliers_iqr)} outliers ({len(outliers_iqr)/len(df)*100:.2f}%)")

# Method 2: Z-Score Method (assumes normality)
z_scores = np.abs(stats.zscore(df[feature_col].dropna()))
outliers_z = df[feature_col].dropna()[z_scores > 3]
print(f"Z-Score Method (|z|>3): {len(outliers_z)} outliers")

# Method 3: Modified Z-Score (robust, uses median)
median = df[feature_col].median()
mad = np.median(np.abs(df[feature_col] - median))
modified_z = 0.6745 * (df[feature_col] - median) / mad
outliers_modified_z = df[np.abs(modified_z) > 3.5]
print(f"Modified Z-Score Method: {len(outliers_modified_z)} outliers")
```

**Step 3: Handle outliers**

```python
# Option 1: Cap/Winsorize (recommended for most cases)
from scipy.stats.mstats import winsorize
df[feature_col + '_winsorized'] = winsorize(df[feature_col].fillna(df[feature_col].median()), limits=[0.05, 0.05])

# Option 2: Log transform (good for right-skewed data)
if (df[feature_col] > 0).all():
    df[feature_col + '_log'] = np.log1p(df[feature_col])

# Option 3: Clip to bounds
df[feature_col + '_clipped'] = df[feature_col].clip(lower=lower_bound, upper=upper_bound)

# Option 4: Remove (only if you're SURE they're errors)
df_no_outliers = df[(df[feature_col] >= lower_bound) & (df[feature_col] <= upper_bound)]
print(f"Rows remaining after removal: {len(df_no_outliers)}/{len(df)}")
```

---

### 1.3 Feature Importance Detection

**Step 1: Pearson correlation (linear relationship)**

```python
pearson_corr, p_value = stats.pearsonr(
    df[feature_col].dropna(), 
    df.loc[df[feature_col].notna(), target_col]
)
print(f"Pearson r = {pearson_corr:.4f}, p-value = {p_value:.4e}")
print("Strong" if abs(pearson_corr) > 0.5 else "Moderate" if abs(pearson_corr) > 0.3 else "Weak")
```

**Step 2: Spearman correlation (monotonic, non-linear)**

```python
spearman_corr, p_value = stats.spearmanr(
    df[feature_col].dropna(), 
    df.loc[df[feature_col].notna(), target_col]
)
print(f"Spearman ρ = {spearman_corr:.4f}, p-value = {p_value:.4e}")
```

**Step 3: Mutual Information (captures non-linear dependencies)**

```python
from sklearn.feature_selection import mutual_info_regression

# Requires no NaN
clean_df = df[[feature_col, target_col]].dropna()
mi = mutual_info_regression(clean_df[[feature_col]], clean_df[target_col], random_state=42)
print(f"Mutual Information = {mi[0]:.4f}")
print("Higher MI → stronger dependency (linear or non-linear)")
```

**Step 4: Visual assessment**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter with regression line
sns.regplot(x=feature_col, y=target_col, data=df, ax=axes[0], scatter_kws={'alpha':0.4})
axes[0].set_title(f"Linear Fit: r={pearson_corr:.3f}")

# LOWESS (non-linear trend)
sns.regplot(x=feature_col, y=target_col, data=df, ax=axes[1], lowess=True, 
            scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
axes[1].set_title("LOWESS Smooth")

# Residual plot
from sklearn.linear_model import LinearRegression
clean = df[[feature_col, target_col]].dropna()
model = LinearRegression().fit(clean[[feature_col]], clean[target_col])
residuals = clean[target_col] - model.predict(clean[[feature_col]])
axes[2].scatter(clean[feature_col], residuals, alpha=0.4)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_title("Residual Plot (check for patterns)")

plt.tight_layout()
plt.show()
```

**Decision Framework:**

| Metric | Threshold | Verdict |
|--------|-----------|---------|
| \|Pearson r\| > 0.5 | Strong linear | Keep |
| \|Spearman ρ\| > 0.5 | Strong monotonic | Keep |
| MI > 0.1 | Non-linear signal | Keep |
| p-value < 0.05 | Statistically significant | Keep |
| All low | No signal | Consider dropping |

---

## 2. Categorical Feature vs Numerical Target

> Example: `department` (feature) → `salary` (target)

### 2.1 Handling Missing Data

**Step 1: Detect and visualize**

```python
feature_col = "department"
target_col = "salary"

print(f"Missing in {feature_col}: {df[feature_col].isnull().sum()} ({df[feature_col].isnull().mean()*100:.2f}%)")
print(f"\nValue counts:\n{df[feature_col].value_counts()}")
```

**Step 2: Check if missingness is informative**

```python
# Compare target distribution where feature is missing vs not
missing_target = df.loc[df[feature_col].isnull(), target_col].dropna()
present_target = df.loc[df[feature_col].notna(), target_col].dropna()

if len(missing_target) > 0:
    stat, p_value = stats.mannwhitneyu(missing_target, present_target, alternative='two-sided')
    print(f"Mann-Whitney U p-value: {p_value:.4f}")
    print("Missingness IS informative" if p_value < 0.05 else "Missingness is random (MCAR)")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(missing_target, label=f'{feature_col} is Missing', ax=ax)
    sns.kdeplot(present_target, label=f'{feature_col} is Present', ax=ax)
    ax.set_title(f"Target distribution by missingness in {feature_col}")
    ax.legend()
    plt.show()
```

**Step 3: Imputation strategies**

```python
# Strategy 1: Mode imputation (most common category)
mode_value = df[feature_col].mode()[0]
df[feature_col + '_imputed'] = df[feature_col].fillna(mode_value)

# Strategy 2: "Unknown" category (preserves missingness signal)
df[feature_col + '_imputed_v2'] = df[feature_col].fillna("Unknown")

# Strategy 3: Probabilistic imputation (sample from existing distribution)
value_probs = df[feature_col].value_counts(normalize=True)
n_missing = df[feature_col].isnull().sum()
random_fill = np.random.choice(value_probs.index, size=n_missing, p=value_probs.values)
df.loc[df[feature_col].isnull(), feature_col + '_prob_imputed'] = random_fill
```

---

### 2.2 Detecting and Handling Outliers

> For categorical features, "outliers" = **rare categories**. For the numerical target, use standard outlier detection per group.

**Step 1: Identify rare categories**

```python
# Frequency analysis
value_counts = df[feature_col].value_counts()
freq_pct = df[feature_col].value_counts(normalize=True) * 100

print("Category Frequencies:")
print(pd.DataFrame({'Count': value_counts, 'Percent': freq_pct.round(2)}))

# Rare category threshold (e.g., < 5% or < 10 observations)
rare_threshold_pct = 5
rare_categories = freq_pct[freq_pct < rare_threshold_pct].index.tolist()
print(f"\nRare categories (<{rare_threshold_pct}%): {rare_categories}")
```

**Step 2: Visualize target distribution per category (detect target outliers per group)**

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot per category
sns.boxplot(x=feature_col, y=target_col, data=df, ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_title(f"{target_col} by {feature_col}")

# Violin plot (shows full distribution)
sns.violinplot(x=feature_col, y=target_col, data=df, ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_title(f"Violin: {target_col} by {feature_col}")

plt.tight_layout()
plt.show()
```

**Step 3: Handle rare categories and group-wise outliers**

```python
# Handle rare categories: group into "Other"
min_samples = max(10, int(len(df) * 0.05))
category_counts = df[feature_col].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
df[feature_col + '_grouped'] = df[feature_col].where(df[feature_col].isin(valid_categories), "Other")

# Handle target outliers within each group
def remove_group_outliers(group):
    Q1 = group[target_col].quantile(0.25)
    Q3 = group[target_col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (group[target_col] >= Q1 - 1.5*IQR) & (group[target_col] <= Q3 + 1.5*IQR)
    return group[mask]

df_clean = df.groupby(feature_col, group_keys=False).apply(remove_group_outliers)
print(f"Rows after group-wise outlier removal: {len(df_clean)}/{len(df)}")
```

---

### 2.3 Feature Importance Detection

**Step 1: ANOVA F-test (are group means different?)**

```python
from scipy.stats import f_oneway

# Get groups
groups = [group[target_col].dropna().values for name, group in df.groupby(feature_col)]

# Filter out groups with less than 2 samples
groups = [g for g in groups if len(g) >= 2]

f_stat, p_value = f_oneway(*groups)
print(f"ANOVA F-statistic = {f_stat:.4f}, p-value = {p_value:.4e}")
print("Feature IS important (groups have different means)" if p_value < 0.05 else "Feature is NOT important")
```

**Step 2: Kruskal-Wallis test (non-parametric alternative)**

```python
from scipy.stats import kruskal

h_stat, p_value = kruskal(*groups)
print(f"Kruskal-Wallis H = {h_stat:.4f}, p-value = {p_value:.4e}")
print("Feature IS important" if p_value < 0.05 else "Feature is NOT important")
```

**Step 3: Effect size — Eta-squared (η²)**

```python
# η² = SS_between / SS_total
grand_mean = df[target_col].mean()
group_stats = df.groupby(feature_col)[target_col].agg(['mean', 'count'])

ss_between = ((group_stats['mean'] - grand_mean) ** 2 * group_stats['count']).sum()
ss_total = ((df[target_col] - grand_mean) ** 2).sum()
eta_squared = ss_between / ss_total

print(f"Eta-squared (η²) = {eta_squared:.4f}")
print("Small" if eta_squared < 0.06 else "Medium" if eta_squared < 0.14 else "Large", "effect size")
```

**Step 4: Mutual Information**

```python
from sklearn.feature_selection import mutual_info_regression

# Encode categorical
le = LabelEncoder()
encoded_feature = le.fit_transform(df[feature_col].fillna("Missing"))

mi = mutual_info_regression(encoded_feature.reshape(-1, 1), df[target_col].fillna(df[target_col].median()))
print(f"Mutual Information = {mi[0]:.4f}")
```

**Step 5: Visualization**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean target per category with CI
sns.barplot(x=feature_col, y=target_col, data=df, ax=axes[0], ci=95)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_title(f"Mean {target_col} by {feature_col} (95% CI)")

# Strip plot (individual points)
sns.stripplot(x=feature_col, y=target_col, data=df, ax=axes[1], alpha=0.4, jitter=True)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_title(f"Individual points: {target_col} by {feature_col}")

plt.tight_layout()
plt.show()
```

**Decision Framework:**

| Test | Condition | Threshold | Verdict |
|------|-----------|-----------|---------|
| ANOVA / Kruskal-Wallis | p-value | < 0.05 | Keep |
| Eta-squared | Effect size | > 0.06 | Keep |
| Mutual Information | Dependency | > 0.05 | Keep |
| Visual | Means differ clearly | — | Keep |

---

## 3. Numerical Feature vs Categorical Target

> Example: `income` (feature) → `loan_default` (Yes/No target)

### 3.1 Handling Missing Data

**Step 1: Assess missing data per class**

```python
feature_col = "income"
target_col = "loan_default"

# Check if missingness differs by target class
missing_by_class = df.groupby(target_col)[feature_col].apply(lambda x: x.isnull().mean() * 100)
print(f"Missing % per class:\n{missing_by_class}")

# Chi-square test: is missingness independent of target?
contingency = pd.crosstab(df[feature_col].isnull(), df[target_col])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi2 test for missingness vs target: p={p_value:.4f}")
print("Missingness IS related to target (MAR)" if p_value < 0.05 else "MCAR likely")
```

**Step 2: Class-conditional imputation**

```python
# Strategy 1: Impute with class-specific median (preserves class separation)
class_medians = df.groupby(target_col)[feature_col].median()
print(f"Median per class:\n{class_medians}")

df[feature_col + '_imputed'] = df[feature_col].copy()
for cls in df[target_col].dropna().unique():
    mask = (df[target_col] == cls) & (df[feature_col].isnull())
    df.loc[mask, feature_col + '_imputed'] = class_medians[cls]

# Strategy 2: For remaining NaN (target also missing), use global median
df[feature_col + '_imputed'].fillna(df[feature_col].median(), inplace=True)

# Strategy 3: Add missingness flag (useful when missingness is informative)
df[feature_col + '_missing_flag'] = df[feature_col].isnull().astype(int)
```

---

### 3.2 Detecting and Handling Outliers

**Step 1: Detect outliers per class**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot per class
sns.boxplot(x=target_col, y=feature_col, data=df, ax=axes[0])
axes[0].set_title(f"{feature_col} distribution by {target_col}")

# Overlapping histograms
for cls in df[target_col].dropna().unique():
    subset = df[df[target_col] == cls][feature_col].dropna()
    axes[1].hist(subset, bins=30, alpha=0.5, label=f"{target_col}={cls}", density=True)
axes[1].legend()
axes[1].set_title(f"{feature_col} distribution per class")

plt.tight_layout()
plt.show()
```

**Step 2: Class-aware outlier detection**

```python
outlier_report = {}
for cls in df[target_col].dropna().unique():
    subset = df[df[target_col] == cls][feature_col].dropna()
    Q1, Q3 = subset.quantile(0.25), subset.quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = ((subset < Q1 - 1.5*IQR) | (subset > Q3 + 1.5*IQR)).sum()
    outlier_report[cls] = {'count': n_outliers, 'percent': n_outliers/len(subset)*100}

print("Outliers per class:")
print(pd.DataFrame(outlier_report).T)
```

**Step 3: Handle with care — don't remove class-separating "outliers"**

```python
# IMPORTANT: If one class has naturally higher/lower values, those aren't outliers!
# Use Isolation Forest (unsupervised, class-aware)
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.05, random_state=42)
clean_data = df[[feature_col]].dropna()
outlier_labels = clf.fit_predict(clean_data)

df.loc[clean_data.index, 'is_outlier'] = (outlier_labels == -1).astype(int)
print(f"Isolation Forest outliers: {(outlier_labels == -1).sum()}")

# Check if "outliers" are actually one class — if so, DON'T remove them
outlier_class_dist = df[df['is_outlier'] == 1][target_col].value_counts(normalize=True)
print(f"\nClass distribution among 'outliers':\n{outlier_class_dist}")
print("\n⚠️ If outliers are dominated by one class, they're SIGNAL, not noise!")
```

---

### 3.3 Feature Importance Detection

**Step 1: Two-sample test (binary target)**

```python
# For binary target
classes = df[target_col].dropna().unique()

if len(classes) == 2:
    group1 = df[df[target_col] == classes[0]][feature_col].dropna()
    group2 = df[df[target_col] == classes[1]][feature_col].dropna()
    
    # Check normality
    _, p_normal_1 = stats.shapiro(group1.sample(min(500, len(group1))))
    _, p_normal_2 = stats.shapiro(group2.sample(min(500, len(group2))))
    
    if p_normal_1 > 0.05 and p_normal_2 > 0.05:
        # Parametric: t-test
        stat, p_value = stats.ttest_ind(group1, group2)
        print(f"T-test: t={stat:.4f}, p={p_value:.4e}")
    else:
        # Non-parametric: Mann-Whitney U
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        print(f"Mann-Whitney U: stat={stat:.4f}, p={p_value:.4e}")
    
    # Effect size: Cohen's d
    cohens_d = (group1.mean() - group2.mean()) / np.sqrt(
        ((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2)
    )
    print(f"Cohen's d = {cohens_d:.4f}")
    print("Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large", "effect")
```

**Step 2: Multi-class — ANOVA or Kruskal-Wallis**

```python
groups = [group[feature_col].dropna().values for name, group in df.groupby(target_col)]
groups = [g for g in groups if len(g) >= 2]

h_stat, p_value = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_value:.4e}")
```

**Step 3: Mutual Information for Classification**

```python
from sklearn.feature_selection import mutual_info_classif

clean = df[[feature_col, target_col]].dropna()
le = LabelEncoder()
y_encoded = le.fit_transform(clean[target_col])

mi = mutual_info_classif(clean[[feature_col]], y_encoded, random_state=42)
print(f"Mutual Information = {mi[0]:.4f}")
```

**Step 4: ROC-AUC as a univariate importance score (binary target)**

```python
from sklearn.metrics import roc_auc_score

if len(classes) == 2:
    clean = df[[feature_col, target_col]].dropna()
    y_binary = (clean[target_col] == classes[1]).astype(int)
    
    # AUC of using raw feature as "prediction"
    auc = roc_auc_score(y_binary, clean[feature_col])
    auc = max(auc, 1 - auc)  # Flip if negatively correlated
    print(f"Univariate AUC = {auc:.4f}")
    print("Useful" if auc > 0.6 else "Weak" if auc > 0.55 else "Useless")
```

**Step 5: Visualization**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# KDE per class
for cls in classes:
    subset = df[df[target_col] == cls][feature_col].dropna()
    sns.kdeplot(subset, label=f"{target_col}={cls}", ax=axes[0])
axes[0].set_title(f"Density of {feature_col} by class")
axes[0].legend()

# Boxplot
sns.boxplot(x=target_col, y=feature_col, data=df, ax=axes[1])
axes[1].set_title(f"Boxplot: {feature_col} by {target_col}")

# Cumulative distribution
for cls in classes:
    subset = df[df[target_col] == cls][feature_col].dropna().sort_values()
    axes[2].plot(subset, np.linspace(0, 1, len(subset)), label=f"{target_col}={cls}")
axes[2].set_title("CDF per class (more separation = more important)")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## 4. Categorical Feature vs Categorical Target

> Example: `education_level` (feature) → `loan_default` (Yes/No target)

### 4.1 Handling Missing Data

**Step 1: Assess missingness**

```python
feature_col = "education_level"
target_col = "loan_default"

print(f"Missing {feature_col}: {df[feature_col].isnull().sum()} ({df[feature_col].isnull().mean()*100:.2f}%)")
print(f"Missing {target_col}: {df[target_col].isnull().sum()}")

# Cross-tab of missingness
print("\nMissingness pattern:")
print(pd.crosstab(df[feature_col].isnull(), df[target_col], margins=True))
```

**Step 2: Test if missingness is related to target**

```python
contingency = pd.crosstab(df[feature_col].isnull(), df[target_col])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi2 test (missingness vs target): χ²={chi2:.4f}, p={p_value:.4f}")
print("Missingness IS informative" if p_value < 0.05 else "MCAR")
```

**Step 3: Imputation strategies**

```python
# Strategy 1: Mode
df[feature_col + '_imputed'] = df[feature_col].fillna(df[feature_col].mode()[0])

# Strategy 2: "Missing" as its own category (BEST if missingness is informative)
df[feature_col + '_imputed_v2'] = df[feature_col].fillna("Missing")

# Strategy 3: Class-conditional mode
class_modes = df.groupby(target_col)[feature_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
df[feature_col + '_class_imputed'] = df[feature_col].copy()
for cls in df[target_col].dropna().unique():
    mask = (df[target_col] == cls) & (df[feature_col].isnull())
    df.loc[mask, feature_col + '_class_imputed'] = class_modes[cls]
```

---

### 4.2 Detecting and Handling Outliers

> "Outliers" in categorical-categorical = **rare combinations** or **unexpected proportions**

**Step 1: Identify rare categories and combinations**

```python
# Rare categories
print("Feature value counts:")
print(df[feature_col].value_counts())
print(f"\nTarget value counts:")
print(df[target_col].value_counts())

# Rare combinations (cells in contingency table with very low counts)
contingency_full = pd.crosstab(df[feature_col], df[target_col])
print("\nContingency table:")
print(contingency_full)

# Flag cells with fewer than 5 expected observations
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_full)
low_expected = (expected < 5).sum()
print(f"\nCells with expected count < 5: {low_expected}/{expected.size}")
print("⚠️ Chi-square test may be unreliable!" if low_expected > 0 else "✓ All cells adequate")
```

**Step 2: Visualize**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Stacked bar chart (proportions)
ct_norm = pd.crosstab(df[feature_col], df[target_col], normalize='index')
ct_norm.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title(f"Target proportion by {feature_col}")
axes[0].set_ylabel("Proportion")
axes[0].legend(title=target_col)

# Heatmap of residuals (observed - expected)
residuals = contingency_full.values - expected
sns.heatmap(pd.DataFrame(residuals, index=contingency_full.index, columns=contingency_full.columns), 
            annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=axes[1])
axes[1].set_title("Residuals (Observed - Expected)\nRed = more than expected")

plt.tight_layout()
plt.show()
```

**Step 3: Handle rare categories**

```python
# Group rare categories
min_count = max(10, int(len(df) * 0.03))  # At least 3% or 10 samples
rare_cats = df[feature_col].value_counts()[df[feature_col].value_counts() < min_count].index
df[feature_col + '_grouped'] = df[feature_col].replace(rare_cats, "Other")

print(f"Grouped {len(rare_cats)} rare categories into 'Other'")
print(df[feature_col + '_grouped'].value_counts())
```

---

### 4.3 Feature Importance Detection

**Step 1: Chi-Square test of independence**

```python
contingency = pd.crosstab(df[feature_col].fillna("Missing"), df[target_col])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"Chi-Square test: χ²={chi2:.4f}, dof={dof}, p={p_value:.4e}")
print("Feature IS important (variables are dependent)" if p_value < 0.05 else "Feature NOT important")
```

**Step 2: Cramér's V (effect size for chi-square)**

```python
n = contingency.sum().sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Cramér's V = {cramers_v:.4f}")
print("Negligible" if cramers_v < 0.1 else "Small" if cramers_v < 0.3 else "Medium" if cramers_v < 0.5 else "Large")
```

**Step 3: Theil's Uncertainty Coefficient (asymmetric — feature predicting target)**

```python
from scipy.stats import entropy

def theils_u(x, y):
    """How much does knowing x reduce uncertainty about y?"""
    contingency = pd.crosstab(x, y)
    # H(Y)
    y_probs = contingency.sum(axis=0) / contingency.sum().sum()
    h_y = entropy(y_probs)
    
    # H(Y|X)
    x_totals = contingency.sum(axis=1)
    h_y_given_x = 0
    for i in range(len(x_totals)):
        p_x = x_totals.iloc[i] / contingency.sum().sum()
        row_probs = contingency.iloc[i] / x_totals.iloc[i]
        h_y_given_x += p_x * entropy(row_probs)
    
    if h_y == 0:
        return 0
    return (h_y - h_y_given_x) / h_y

u = theils_u(df[feature_col].fillna("Missing"), df[target_col].fillna("Missing"))
print(f"Theil's U (feature→target) = {u:.4f}")
print("0 = feature tells nothing about target, 1 = perfectly predicts target")
```

**Step 4: Mutual Information**

```python
from sklearn.feature_selection import mutual_info_classif

le_feature = LabelEncoder()
le_target = LabelEncoder()

clean = df[[feature_col, target_col]].dropna()
X_encoded = le_feature.fit_transform(clean[feature_col]).reshape(-1, 1)
y_encoded = le_target.fit_transform(clean[target_col])

mi = mutual_info_classif(X_encoded, y_encoded, discrete_features=True, random_state=42)
print(f"Mutual Information = {mi[0]:.4f}")
```

**Step 5: Visualization**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Mosaic-style: proportional bar chart
ct = pd.crosstab(df[feature_col], df[target_col], normalize='index')
ct.plot(kind='bar', ax=axes[0], colormap='Set2')
axes[0].set_title(f"Target rate by {feature_col}")
axes[0].set_ylabel("Proportion")
axes[0].axhline(df[target_col].value_counts(normalize=True).iloc[0], 
                color='red', linestyle='--', label='Overall rate')
axes[0].legend()

# Heatmap of contingency table
sns.heatmap(pd.crosstab(df[feature_col], df[target_col]), annot=True, fmt='d', 
            cmap='YlOrRd', ax=axes[1])
axes[1].set_title("Counts per combination")

# Standardized residuals heatmap
std_residuals = (contingency.values - expected) / np.sqrt(expected)
sns.heatmap(pd.DataFrame(std_residuals, index=contingency.index, columns=contingency.columns),
            annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[2])
axes[2].set_title("Standardized Residuals\n(|val|>2 = significant)")

plt.tight_layout()
plt.show()
```

**Decision Framework:**

| Test | Threshold | Verdict |
|------|-----------|---------|
| Chi-square p-value | < 0.05 | Significant association |
| Cramér's V | > 0.1 | Meaningful effect |
| Theil's U | > 0.05 | Feature reduces target uncertainty |
| Mutual Information | > 0.02 | Non-trivial dependency |

---

## 5. Quick Reference Cheat Sheet

| Scenario | Missing Data | Outliers | Importance Test |
|----------|-------------|----------|----------------|
| Num → Num | Median/KNN impute | IQR/Z-score + Winsorize | Pearson, Spearman, MI |
| Cat → Num | Mode/"Unknown" | Rare categories → "Other" | ANOVA, η², MI |
| Num → Cat | Class-conditional median | Class-aware IQR, Isolation Forest | Mann-Whitney, Cohen's d, AUC, MI |
| Cat → Cat | Mode/"Missing" category | Rare combos → group | Chi-square, Cramér's V, Theil's U |

---

## 6. Golden Rules

1. **Always visualize first** — numbers lie, pictures don't.
2. **Check missingness type BEFORE imputing** — wrong imputation introduces bias.
3. **Outliers aren't always bad** — they might be your most important signal.
4. **Use multiple importance metrics** — one test can miss non-linear relationships.
5. **Domain knowledge > statistics** — a p-value of 0.06 with a known causal mechanism is still useful.
6. **Document your decisions** — future-you will thank present-you.

---

This is the complete guide. Would you like me to save this as a `.md` file in your workspace?
