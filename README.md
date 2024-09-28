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
