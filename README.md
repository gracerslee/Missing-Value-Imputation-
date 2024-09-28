# Missing Value Imputation using Various Methods

This project demonstrates a step-by-step approach to handle missing values in a dataset using various imputation techniques and compares their performances based on Root Mean Square Error (RMSE) or variance deviation.

## Step-by-step Implementation

### 1. Generate a Complete DataFrame and a DataFrame with Missing Values

We start by creating two DataFrames using `pandas` and `numpy`:
- One with complete values
- One with randomly introduced missing values

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a complete DataFrame (1000 rows, 10 columns)
complete_df = pd.DataFrame(np.random.randn(1000, 10), columns=[f'col_{i}' for i in range(1, 11)])

# Introduce missing values randomly (second DataFrame)
missing_df = complete_df.copy()
missingness_prop = 0.1  # 10% missing values
missing_mask = np.random.rand(*missing_df.shape) < missingness_prop
missing_df[missing_mask] = np.nan
```

### 2. Visualize Missing Data
We visualize the missing data in the missing_df using a heatmap:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize missing data with a density plot
sns.heatmap(missing_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

### 3. Perform Imputation using Various Algorithms
We will apply different imputation methods to handle missing values:

1. Mean Imputation
2. Median Imputation
3. Frequent Value Imputation
4. Zero Imputation
5. Constant Imputation (e.g., custom values)
6. K-NN Imputation
7. MICE (Multivariate Imputation by Chained Equations)
We will apply different imputation methods to handle missing values:

```python
!pip install scikit-learn fancyimpute tensorflow
```
Below is the implementation of the imputation methods:

from sklearn.impute import SimpleImputer, KNNImputer
from fancyimpute import IterativeImputer  # For MICE

# Imputation functions
def mean_imputation(df):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def median_imputation(df):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def most_frequent_imputation(df):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def zero_imputation(df):
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def constant_imputation(df, fill_value):
    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def knn_imputation(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def mice_imputation(df):
    imputer = IterativeImputer()
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

### 4. Compare and Evaluate the Performance of Algorithms
To compare the performance of the imputation methods, we calculate the RMSE between the imputed data and the original data:


