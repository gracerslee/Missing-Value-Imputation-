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
```python
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
```

### 4. Compare and Evaluate the Performance of Algorithms
To compare the performance of the imputation methods, we calculate the RMSE between the imputed data and the original data:
```python
from sklearn.metrics import mean_squared_error

def evaluate_imputation(original_df, imputed_df):
    # Only compare non-missing values to the original data
    mask = original_df.notna()
    mse = mean_squared_error(original_df[mask], imputed_df[mask])
    return np.sqrt(mse)
```
Apply the imputation methods and compute the RMSE:
```python
methods = {
    'Mean Imputation': mean_imputation,
    'Median Imputation': median_imputation,
    'Most Frequent Value Imputation': most_frequent_imputation,
    'Zero Imputation': zero_imputation,
    'Constant Imputation (e.g., -1)': lambda df: constant_imputation(df, -1),
    'K-NN Imputation': knn_imputation,
    'MICE Imputation': mice_imputation,
}

results = {}
for method_name, imputation_func in methods.items():
    imputed_df = imputation_func(missing_df)
    rmse = evaluate_imputation(complete_df, imputed_df)
    results[method_name] = rmse

# Find the best method
best_method = min(results, key=results.get)
print(f"Best imputation method: {best_method} with RMSE: {results[best_method]}")
```
### 5. Visualize the Results 
The RMSE scores of each method can be visualized using a bar plot:
```python
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('RMSE of Different Imputation Methods')
plt.xticks(rotation=45)
plt.ylabel('RMSE')
plt.show()
```
### 6. Export the Best Imputed Data 
After determining the best imputation method, the imputed DataFrame can be exported as a CSV file:
```python
best_imputed_df = methods[best_method](missing_df)
best_imputed_df.to_csv('best_imputed_data.csv', index=False)
print("Best imputed data exported to 'best_imputed_data.csv'.")
```
### 7. Wrap Everything in a Single Function 
Finally, all the steps are wrapped into a single function for easy reuse:
```python
def impute_and_evaluate(df, complete_df):
    # Visualize missing values
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    # Dictionary of imputation methods
    methods = {
        'Mean Imputation': mean_imputation,
        'Median Imputation': median_imputation,
        'Most Frequent Value Imputation': most_frequent_imputation,
        'Zero Imputation': zero_imputation,
        'Constant Imputation (-1)': lambda df: constant_imputation(df, -1),
        'K-NN Imputation': knn_imputation,
        'MICE Imputation': mice_imputation,
    }

    # Evaluate methods
    results = {}
    for method_name, imputation_func in methods.items():
        imputed_df = imputation_func(df)
        rmse = evaluate_imputation(complete_df, imputed_df)
        results[method_name] = rmse

    # Find the best method
    best_method = min(results, key=results.get)
    print(f"Best imputation method: {best_method} with RMSE: {results[best_method]}")

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('RMSE of Different Imputation Methods')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    plt.show()

    # Export the best imputed data
    best_imputed_df = methods[best_method](df)
    best_imputed_df.to_csv('best_imputed_data.csv', index=False)
    print("Best imputed data exported to 'best_imputed_data.csv'.")

impute_and_evaluate(missing_df, complete_df)
```
### Notes:
1. DNN Imputation: This method can be complex and is typically not necessary for basic missing value imputation.
2. Hot-Deck Imputation: Requires custom implementation based on similar rows.

### License
MIT License 
```vbnet
This README file provides clear step-by-step instructions using Markdown formatting to document the Python code, visualization, and imputation methods.
```








