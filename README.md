# Crash Courses

Crash Courses is a growing, notebook-based data-science curriculum built to turn concepts into practical judgment. It is a place to relearn core data-science foundations, prepare for interviews, sharpen Python and SQL skills, and build a portfolio of work that explains both the implementation and the decisions behind it.

The notebooks are intentionally text-heavy. Each one should make the reasoning visible: what problem is being solved, why a method fits, how the result is evaluated, where it can fail, and what someone should do with the result.

## Goals

- Build strong foundations in data preparation, modeling, evaluation, statistics, and experimentation.
- Practice the SQL, Python, probability, and product-thinking skills common in data-science interviews.
- Learn modern ML systems through RAG, LLM evaluation, explainability, and responsible AI.
- Produce a small number of decision-ready case studies that demonstrate technical depth and clear communication.
- Apply reproducible engineering practices as the projects become portfolio-ready.

## Start Here

Start with [Unit 0: Foundations & Models](unit0_foundations_and_models/README.md). Then move to [Unit 1: Modern ML & Responsible AI](unit1_modern_ml_and_responsible_ai/README.md) to build the highest-leverage portfolio pieces. Complete either the churn or forecasting project in [Unit 3: Applied Case Studies](unit3_applied_case_studies/README.md) before trying to finish the full curriculum.

## Table of Contents

### Unit 0: Foundations & Models

| Notebook | Focus |
| --- | --- |
| [Preprocessing](unit0_foundations_and_models/preprocessing.ipynb) | Missing values, encoding, scaling, transformations, and leakage-safe pipelines. |
| [Supervised Learning](unit0_foundations_and_models/supervised_learning.ipynb) | Regression and classification with the housing-prices dataset. |
| [Model Evaluation & Error Analysis](unit0_foundations_and_models/model_evaluation_error_analysis.ipynb) | Baselines, validation, metrics, thresholds, calibration, and error slices. |
| [Unsupervised Learning](unit0_foundations_and_models/unsupervised_learning.ipynb) | Finding structure in unlabeled data. |

### Unit 1: Modern ML & Responsible AI

| Notebook | Focus |
| --- | --- |
| [RAG from Scratch](unit1_modern_ml_and_responsible_ai/rag_from_scratch.ipynb) | Ingestion, chunking, embeddings, retrieval, generation, and failure modes. |
| [LLM Evaluation](unit1_modern_ml_and_responsible_ai/llm_evaluation.ipynb) | Evaluation sets, retrieval quality, groundedness, relevance, latency, and cost. |
| [Model Explainability](unit1_modern_ml_and_responsible_ai/model_explainability.ipynb) | Global and local explanations, importance, partial dependence, and SHAP. |
| [Responsible AI & Fairness](unit1_modern_ml_and_responsible_ai/responsible_ai_fairness.ipynb) | Bias, proxy variables, subgroup evaluation, privacy, and safeguards. |

### Unit 2: DS Interview Fundamentals

| Notebook | Focus |
| --- | --- |
| [SQL for Data Science](unit2_ds_interview_fundamentals/sql_for_data_science.ipynb) | Joins, CTEs, window functions, cohorts, funnels, and data-quality checks. |
| [Statistics & Probability](unit2_ds_interview_fundamentals/statistics_probability.ipynb) | Distributions, inference, hypothesis tests, and common reasoning traps. |
| [A/B Testing & Experiment Design](unit2_ds_interview_fundamentals/ab_testing_experiment_design.ipynb) | Metrics, power, effect size, significance, and experimental failure modes. |
| [Time Series & Forecasting](unit2_ds_interview_fundamentals/time_series_forecasting.ipynb) | Chronological validation, baselines, trend, seasonality, and uncertainty. |
| [Python Coding Patterns](unit2_ds_interview_fundamentals/python_coding_patterns.ipynb) | Complexity and reusable interview patterns for data-oriented coding problems. |

### Unit 3: Applied Case Studies

| Project | Focus |
| --- | --- |
| [Traditional DS Case Study](unit3_applied_case_studies/traditional_ds_case_study/) | Choose customer churn or demand forecasting; show a baseline, validation, tuning, error analysis, and business recommendation. |
| [Resume-Job Match RAG](unit3_applied_case_studies/resume_job_match_rag/) | Apply the RAG and evaluation units to a grounded resume and job-description retrieval system. |

### Unit 4: ML System Design

| Notebook | Focus |
| --- | --- |
| [ML System Design Exercises](unit4_ml_system_design/ml_system_design_exercises.ipynb) | Targets, data sources, metrics, inference modes, monitoring, drift, retraining, and rollback. |

### Unit 5: Engineering Practices

Engineering practices are applied throughout the repository: clear project documentation, reproducible environments, reusable code where warranted, focused tests, configuration, and saved pipelines. See [Unit 5: Engineering Practices](unit5_engineering_practices/README.md).

## Learning Approach

Each notebook should follow a consistent shape:

1. State the business or analytical question.
2. Explain the data, assumptions, and quality concerns.
3. Establish a baseline before introducing a more complex approach.
4. Evaluate results with a metric and validation strategy that fit the decision.
5. Analyze errors, limitations, and risks.
6. End with a recommendation, next step, or interview-style reflection.

## Current Data

The supervised and unsupervised learning notebooks use the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). A local copy is available at [unit0_foundations_and_models/datasets/Housing.csv](unit0_foundations_and_models/datasets/Housing.csv), and the notebooks can also download the dataset with `kagglehub`.

## Running the Notebooks

Install the dependencies required by the existing housing-model notebooks:

```bash
pip install kagglehub pandas matplotlib scikit-learn statsmodels
```

Open a notebook in VS Code or Jupyter and run its cells in order. Dependencies and setup instructions for new case studies will be added with those projects.
