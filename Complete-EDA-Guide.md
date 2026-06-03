
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

