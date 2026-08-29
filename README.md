# Crash Courses

Crash Courses is a growing collection of applied, notebook-based introductions to computer science concepts. Each course is designed to give a practical, approachable foundation in a focused topic without requiring a full-length curriculum.

The current focus is data science and machine learning. Rather than introducing each model with a different dataset, the notebooks reuse a shared housing-prices dataset and build from one idea to the next. That makes it easier to compare what each model is designed to do, how its predictions differ, and which tradeoffs matter in a practical setting.

## Learning Approach

The notebooks are intentionally text-heavy. The detailed explanations are there to make each concept approachable and to encourage you to pause at every step: understand what each line of code does, why it belongs in the workflow, and how it affects the model's result.

## Contents

### Machine Learning

| Topic | Notebook | What it covers |
| --- | --- | --- |
| Supervised learning | [Supervised Housing Prices](supervised_housing_prices.ipynb) | Uses housing data to introduce predictive models, compare their behavior on related problems, and explore classification tradeoffs. |
| Unsupervised learning | [Unsupervised Housing Prices](unsupervised_housing_prices.ipynb) | Uses the same housing dataset to introduce finding patterns without predefined labels or a target variable. |

#### Supervised Learning Models

The current notebook works through these models in sequence:

1. **Linear regression**: predicts a continuous value by modeling the relationship between house area and price.
2. **Logistic regression**: turns the housing data into a binary classification problem by predicting whether a house is above a luxury-price threshold.
3. **Decision trees**: visualizes rule-based classification, feature engineering, class imbalance, and the precision-recall tradeoff.
4. **Random forests**: combines many decision trees to compare ensemble behavior with a single tree on the same classification task.

Along the way, the notebook also introduces train/test splits, regression summaries, predicted probabilities, confusion matrices, accuracy, precision, recall, F1 score, feature importance, overfitting, class imbalance, and data leakage.

#### Unsupervised Learning Models

The unsupervised notebook introduces the shift from predicting a known target to finding structure in feature data without ground-truth labels. It outlines four major families of unsupervised models:

1. **Clustering**: groups similar observations, such as with K-Means or DBSCAN.
2. **Dimensionality reduction**: represents high-dimensional data with fewer features, such as with PCA or t-SNE.
3. **Association-rule learning**: finds relationships and co-occurrence patterns, such as with Apriori or Eclat.
4. **Anomaly detection**: identifies unusual observations, such as with Isolation Forest or One-Class SVM.

## Data

The current notebook downloads the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) with `kagglehub`. A local `Housing.csv` is also included for convenience.

## Running the Notebooks

Install the Python packages used by the notebook:

```bash
pip install kagglehub pandas matplotlib scikit-learn statsmodels
```

Open [Supervised Housing Prices](supervised_housing_prices.ipynb) or [Unsupervised Housing Prices](unsupervised_housing_prices.ipynb) in Jupyter or VS Code and run the cells in order.

## Future Topics

This repository will grow beyond machine learning with additional applied crash courses in computer science. New notebooks and sections will be added to this table of contents as they are introduced.
