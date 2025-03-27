# Missing Value Imputation with Traditional and Deep Learning Methods

This project presents a comprehensive evaluation of multiple missing value imputation techniques, ranging from traditional statistical methods to deep learning approaches like Transformers and AutoEncoders. The performance of each method is assessed using metrics such as mean deviation, variance deviation, PCA, t-SNE, and UMAP distances.

## Steps Overview

### 1. Generate Complete and Incomplete DataFrames

```python
# Complete DataFrame with 1000 rows and 10 columns
complete_df = pd.DataFrame(np.random.randn(1000, 10), columns=[f'col_{i}' for i in range(1, 11)])

# Introduce 10% missing values randomly
missing_df = complete_df.copy()
missing_mask = np.random.rand(*missing_df.shape) < 0.1
missing_df[missing_mask] = np.nan
```

### 2. Visualize Missingness

```python
sns.heatmap(missing_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

### 3. Imputation Methods
🧮 Traditional Methods:
* Mean Imputation
* Median Imputation
* Most Frequent Value
* Zero Imputation
* Constant Value (e.g., -1)
* K-NN Imputation
* MICE (Multivariate Imputation by Chained Equations)

🤖 Deep Learning Methods:
* Transformer-based Imputation (Custom Transformer Encoder)
* AutoEncoder-based Imputation
                                                     
Each method is implemented and applied using ImputationEvaluator.
      
### 4. Evaluation Metrics
For each imputation method, the following metrics are computed:                                                          
* **Mean Deviation**: Average difference in feature-wise means between original and imputed data.
* **Variance Deviation**: Average difference in variances.
* **PCA Distance**: L2 distance in 2D PCA-transformed space.
* **t-SNE Distance**: L2 distance in 2D t-SNE space.
* **UMAP Distance**: L2 distance in 2D UMAP space.

### 5. Visualization and Reporting
* Bar plot comparing scaled inverse scores across all metrics.
* PDF report generated using fpdf, including:
  * Missingness heatmap
  * Performance comparison plot
  * Text summary of all metrics
* Best method automatically selected based on overall score average.
                                                             
```python
evaluator = ImputationEvaluator(missing_df)
evaluator.evaluate()
```
### 6. Output Files
Upon running, the following files will be saved in the report/ directory:
* missing_heatmap.png: Visual representation of missing data
* combined_barplot.png: Performance comparison of all methods
* imputation_report.txt: Text report summarizing metrics
* imputation_summary.pdf: Comprehensive PDF report
* best_method_comparison.png: Side-by-side visualization (Original vs Imputed)

✅ Best Method Selection
The best-performing imputation method is selected based on the lowest average score across all five evaluation metrics. Final visual comparison of the original vs. imputed data is generated using PCA, t-SNE, and UMAP projections.

                                                              
### Notes:
1. Futher plan
* Speed Up
* Add new algorithm
* Optimize structure                                                              

### License
MIT License 
```vbnet
This README file provides clear step-by-step instructions using Markdown formatting to document the Python code, visualization, and imputation methods.
```
