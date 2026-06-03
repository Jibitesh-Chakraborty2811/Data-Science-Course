
---

# EDA Survival Guide — Keep It Simple

**Rule: Do ONE thing well at each step. Don't overthink it.**

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("your_data.csv")
```

---

## 1. Numerical Feature → Numerical Target

> e.g., `age` → `salary`

### Missing Data

```python
feature = "age"
target = "salary"

# Check
print(df[[feature, target]].isnull().sum())

# Fix: use median (safe default, handles skew)
df[feature].fillna(df[feature].median(), inplace=True)
```

### Outliers

```python
# Detect with IQR
Q1, Q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

# Fix: clip to bounds (don't delete rows)
df[feature] = df[feature].clip(lower, upper)

# Visualize
sns.boxplot(x=df[feature])
plt.show()
```

### Is the feature useful?

```python
# Correlation — one number tells you a lot
corr, p = stats.spearmanr(df[feature], df[target])
print(f"Spearman correlation: {corr:.3f}, p-value: {p:.4f}")

# Visualize
sns.scatterplot(x=feature, y=target, data=df, alpha=0.5)
plt.show()
```

**Keep if:** |correlation| > 0.3 AND p < 0.05

---

## 2. Categorical Feature → Numerical Target

> e.g., `department` → `salary`

### Missing Data

```python
feature = "department"
target = "salary"

# Fix: just make "Missing" its own category
df[feature].fillna("Unknown", inplace=True)
```

### Outliers

```python
# Rare categories = noise. Group anything with < 5% into "Other"
threshold = 0.05 * len(df)
counts = df[feature].value_counts()
rare = counts[counts < threshold].index
df[feature] = df[feature].replace(rare, "Other")

# Visualize
sns.boxplot(x=feature, y=target, data=df)
plt.xticks(rotation=45)
plt.show()
```

### Is the feature useful?

```python
# Do the groups have different averages?
groups = [g[target].values for _, g in df.groupby(feature)]
stat, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis p-value: {p:.4f}")

# Visualize — this tells you everything
sns.barplot(x=feature, y=target, data=df)
plt.xticks(rotation=45)
plt.show()
```

**Keep if:** p < 0.05 AND bars look visibly different

---

## 3. Numerical Feature → Categorical Target

> e.g., `income` → `loan_default` (Yes/No)

### Missing Data

```python
feature = "income"
target = "loan_default"

# Fix: median
df[feature].fillna(df[feature].median(), inplace=True)
```

### Outliers

```python
# Same IQR clip as before
Q1, Q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
IQR = Q3 - Q1
df[feature] = df[feature].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```

### Is the feature useful?

```python
# Do the classes have different distributions?
classes = df[target].unique()
group1 = df[df[target] == classes[0]][feature]
group2 = df[df[target] == classes[1]][feature]

stat, p = stats.mannwhitneyu(group1, group2)
print(f"Mann-Whitney p-value: {p:.4f}")

# Visualize — overlapping KDEs
for cls in classes:
    sns.kdeplot(df[df[target] == cls][feature], label=str(cls))
plt.legend()
plt.title(f"{feature} by {target}")
plt.show()
```

**Keep if:** p < 0.05 AND the curves are visibly separated

---

## 4. Categorical Feature → Categorical Target

> e.g., `education` → `loan_default` (Yes/No)

### Missing Data

```python
feature = "education"
target = "loan_default"

# Fix: treat missing as its own category
df[feature].fillna("Unknown", inplace=True)
```

### Outliers

```python
# Group rare categories into "Other"
threshold = 0.05 * len(df)
counts = df[feature].value_counts()
rare = counts[counts < threshold].index
df[feature] = df[feature].replace(rare, "Other")
```

### Is the feature useful?

```python
# Chi-square: are the two variables related?
contingency = pd.crosstab(df[feature], df[target])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square p-value: {p:.4f}")

# Cramér's V (strength: 0 = nothing, 1 = perfect)
n = len(df)
v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"Cramér's V: {v:.3f}")

# Visualize
pd.crosstab(df[feature], df[target], normalize='index').plot(kind='bar', stacked=True)
plt.title(f"{target} rate by {feature}")
plt.ylabel("Proportion")
plt.show()
```

**Keep if:** p < 0.05 AND Cramér's V > 0.1

---

## Cheat Sheet (tape this to your monitor)

| Scenario | Fix Missing | Fix Outliers | Test Importance |
|----------|------------|-------------|----------------|
| Num → Num | Median | IQR clip | Spearman corr |
| Cat → Num | "Unknown" | Group rare → "Other" | Kruskal-Wallis |
| Num → Cat | Median | IQR clip | Mann-Whitney |
| Cat → Cat | "Unknown" | Group rare → "Other" | Chi-square + Cramér's V |

**Universal rule:** Always plot it. If you can't see it, it's probably not useful.

---

Here's a clean add-on section for ML-based feature importance:

---

## 5. Feature Importance Using ML Models

> Your colleague is right — decision trees (and their ensembles) naturally rank features by how useful they are for splitting data.

### For Regression (Numerical Target)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

feature_cols = ["age", "income", "experience", "dept_encoded"]  # all your features
target = "salary"

# Prep: drop rows with NaN in target, fill features
X = df[feature_cols].fillna(0)
y = df[target]

# Train a quick Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
model.fit(X, y)

# Method 1: Built-in feature importance (fast, based on impurity reduction)
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance)

# Plot
importance.plot(kind='barh')
plt.xlabel("Importance (impurity-based)")
plt.title("Feature Importance — Regression")
plt.gca().invert_yaxis()
plt.show()
```

```python
# Method 2: Permutation importance (more reliable, slower)
perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

print(perm_imp)

perm_imp.plot(kind='barh', xerr=perm.importances_std)
plt.xlabel("Permutation Importance (drop in R²)")
plt.title("Permutation Importance — Regression")
plt.gca().invert_yaxis()
plt.show()
```

**Keep if:** importance > 0.05 (built-in) or permutation importance > 0

---

### For Binary Classification

```python
from sklearn.ensemble import RandomForestClassifier

feature_cols = ["age", "income", "experience", "dept_encoded"]
target = "loan_default"  # Yes/No or 0/1

X = df[feature_cols].fillna(0)
y = df[target]

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X, y)

# Built-in importance
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance)

importance.plot(kind='barh')
plt.xlabel("Importance")
plt.title("Feature Importance — Binary Classification")
plt.gca().invert_yaxis()
plt.show()
```

```python
# Permutation importance (use scoring='roc_auc' for binary)
from sklearn.inspection import permutation_importance

perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='roc_auc')
perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

perm_imp.plot(kind='barh', xerr=perm.importances_std)
plt.xlabel("Drop in AUC when feature is shuffled")
plt.title("Permutation Importance — Binary Classification")
plt.gca().invert_yaxis()
plt.show()
```

---

### For Multiclass Classification

```python
from sklearn.ensemble import RandomForestClassifier

feature_cols = ["age", "income", "experience", "dept_encoded"]
target = "customer_segment"  # A, B, C, D...

X = df[feature_cols].fillna(0)
y = df[target]

# Train (same code — sklearn handles multiclass automatically)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X, y)

# Built-in importance (works the same for multiclass)
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance)

importance.plot(kind='barh')
plt.xlabel("Importance")
plt.title("Feature Importance — Multiclass")
plt.gca().invert_yaxis()
plt.show()
```

```python
# Permutation importance (use 'accuracy' or 'f1_weighted' for multiclass)
perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='f1_weighted')
perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

perm_imp.plot(kind='barh', xerr=perm.importances_std)
plt.xlabel("Drop in F1 when feature is shuffled")
plt.title("Permutation Importance — Multiclass")
plt.gca().invert_yaxis()
plt.show()
```

---

### When to Use Which Method

| Method | Pros | Cons |
|--------|------|------|
| Built-in (`feature_importances_`) | Fast, one line | Biased toward high-cardinality features |
| Permutation importance | More honest, model-agnostic | Slower (re-evaluates model many times) |

---

### Quick Single Decision Tree (if you want to SEE the logic)

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Shallow tree — just to see top splits
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# The first splits = most important features
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=feature_cols, class_names=[str(c) for c in tree.classes_],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree — Top splits = most important features")
plt.show()

# Importance from the tree
print(pd.Series(tree.feature_importances_, index=feature_cols).sort_values(ascending=False))
```

> **Pro tip:** The feature at the root of the tree (first split) is almost always the single most important feature. If you only remember one thing, remember this.

---

### Practical Advice

1. **Use Random Forest, not a single tree** — single trees are unstable, forests average out the noise.
2. **Always check permutation importance** if built-in importance looks suspicious (e.g., an ID column ranks high).
3. **Set `max_depth=5`** for importance analysis — you don't need a perfect model, just enough to rank features.
4. **Encode categoricals first** — use `pd.get_dummies()` or `LabelEncoder` before feeding to the model.

```python
# Quick encoding for categorical columns before fitting
df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)
# Then use df_encoded as X
```

---

That's it — plug in your columns, run the code, and the bar chart tells you what matters. Want me to save both guides together as a single file?
