# Crash Courses

Applied, notebook-based explorations of computer science concepts.

This repository starts with machine learning. Rather than introducing each model with a different dataset, the notebooks reuse a shared housing-prices dataset and build from one idea to the next. That makes it easier to compare what each model is designed to do, how its predictions differ, and which tradeoffs matter in a practical setting.

## Contents

### Machine Learning

| Topic | Notebook | What it covers |
| --- | --- | --- |
| Supervised and unsupervised learning | [Supervised Housing Prices](supervised_housing_prices.ipynb) | Uses housing data to introduce supervised learning models, compare their behavior on related problems, and set up the transition to unsupervised learning. |

#### Supervised Learning Models

The current notebook works through these models in sequence:

1. **Linear regression**: predicts a continuous value by modeling the relationship between house area and price.
2. **Logistic regression**: turns the housing data into a binary classification problem by predicting whether a house is above a luxury-price threshold.
3. **Decision trees**: visualizes rule-based classification, feature engineering, class imbalance, and the precision-recall tradeoff.
4. **Random forests**: combines many decision trees to compare ensemble behavior with a single tree on the same classification task.

Along the way, the notebook also introduces train/test splits, regression summaries, predicted probabilities, confusion matrices, accuracy, precision, recall, F1 score, feature importance, overfitting, class imbalance, and data leakage.

#### Unsupervised Learning

The notebook introduces the core idea behind unsupervised learning: finding structure in feature data without a predefined target or ground-truth labels. More unsupervised examples will be added here as the course develops.

## Data

The current notebook downloads the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) with `kagglehub`. A local `Housing.csv` is also included for convenience.

## Running the Notebook

Install the Python packages used by the notebook:

```bash
pip install kagglehub pandas matplotlib scikit-learn statsmodels
```

Then open [Supervised Housing Prices](supervised_housing_prices.ipynb) in Jupyter or VS Code and run the cells in order.

## Future Topics

This repository will grow beyond machine learning with additional applied crash courses in computer science. New notebooks and sections will be added to this table of contents as they are introduced.
