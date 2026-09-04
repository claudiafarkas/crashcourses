# Crash Courses

I made this repository to relearn data-science concepts properly, get sharper for interviews, and have work I can point to when I apply. It is a collection of notebooks where I slow down, work through the details, and explain what I am doing instead of only getting a model to run.

Some topics build on each other, but each folder is meant to be useful on its own. The goal is not to race through a giant curriculum. It is to build a set of projects I understand well enough to explain, defend, and improve.

## Start Here

Start with [Foundations &amp; Models](foundations_and_models/README.md) if you want the modeling workflow. [Modern ML &amp; Responsible AI](modern_ml_reponsible_ai/README.md) and [Applied Case Studies](case_studies/README.md) are the main portfolio pieces; the interview topics are there for deliberate practice alongside them.

## Table of Contents

### Topic: Foundations & Models

| Notebook                                                                                      | Focus                                                                           |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [Preprocessing](foundations_and_models/preprocessing.ipynb)                                           | Missing values, encoding, scaling, transformations, and leakage-safe pipelines. |
| [Supervised Learning](foundations_and_models/supervised_learning.ipynb)                               | Regression and classification with the housing-prices dataset.                  |
| [Model Evaluation &amp; Error Analysis](foundations_and_models/model_evaluation_error_analysis.ipynb) | Baselines, validation, metrics, thresholds, calibration, and error slices.      |
| [Unsupervised Learning](foundations_and_models/unsupervised_learning.ipynb)                           | Finding structure in unlabeled data.                                            |

### Topic: Modern ML & Responsible AI

| Notebook / Application                                                  | Focus                                                                           |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [RAG from Scratch](modern_ml_reponsible_ai/rag_from_scratch.ipynb)                     | Ingestion, chunking, embeddings, retrieval, generation, and failure modes.      |
| [Cloud ☁️ (RAG Assistant)](modern_ml_reponsible_ai/rag_assistant/app.py)                | Localhost Streamlit assistant querying `foundations_and_models` with grounded notebook citations. |
| [LLM Evaluation](modern_ml_reponsible_ai/llm_evaluation.ipynb)                         | Evaluation sets, retrieval quality, groundedness, relevance, latency, and cost. |
| [Model Explainability](modern_ml_reponsible_ai/model_explainability.ipynb)             | Global and local explanations, importance, partial dependence, and SHAP.        |
| [Responsible AI &amp; Fairness](modern_ml_reponsible_ai/responsible_ai_fairness.ipynb) | Bias, proxy variables, subgroup evaluation, privacy, and safeguards.            |

### Topic: DS Interview Fundamentals

| Notebook                                                                                        | Focus                                                                         |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [SQL for Data Science](ds_interview_fundamentals/sql_for_data_science.ipynb)                        | Joins, CTEs, window functions, cohorts, funnels, and data-quality checks.     |
| [Statistics &amp; Probability](ds_interview_fundamentals/statistics_probability.ipynb)              | Distributions, inference, hypothesis tests, and common reasoning traps.       |
| [A/B Testing &amp; Experiment Design](ds_interview_fundamentals/ab_testing_experiment_design.ipynb) | Metrics, power, effect size, significance, and experimental failure modes.    |
| [Time Series &amp; Forecasting](ds_interview_fundamentals/time_series_forecasting.ipynb)            | Chronological validation, baselines, trend, seasonality, and uncertainty.     |
| [Python Coding Patterns](ds_interview_fundamentals/python_coding_patterns.ipynb)                    | Complexity and reusable interview patterns for data-oriented coding problems. |

### Topic: Applied Case Studies

| Project                                                             | Focus                                                                                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [Traditional DS Case Study](case_studies/traditional_ds_case_study/) | Choose customer churn or demand forecasting; show a baseline, validation, tuning, error analysis, and business recommendation. |
| [Resume-Job Match RAG](case_studies/resume_job_match_rag/)           | Apply the RAG and evaluation topics to a grounded resume and job-description retrieval system.                                 |

### Topic: ML System Design

| Notebook                                                                       | Focus                                                                                         |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| [ML System Design Exercises](ml_system_design/ml_system_design_exercises.ipynb) | Targets, data sources, metrics, inference modes, monitoring, drift, retraining, and rollback. |

### Topic: Engineering Practices

This is a set of habits I apply across the repository: clear project documentation, reproducible environments, reusable code where warranted, focused tests, configuration, and saved pipelines. See [Engineering Practices](engineering_practices/README.md).

## Learning Approach

Most notebooks follow a similar shape:

1. State the business or analytical question.
2. Explain the data, assumptions, and quality concerns.
3. Establish a baseline before introducing a more complex approach.
4. Evaluate results with a metric and validation strategy that fit the decision.
5. Analyze errors, limitations, and risks.
6. End with a recommendation, next step, or interview-style reflection.

## Current Data

The supervised and unsupervised learning notebooks use the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). A local copy is available at [foundations_and_models/datasets/Housing.csv](foundations_and_models/datasets/Housing.csv), and the notebooks can also download the dataset with `kagglehub`.

## Running the Notebooks & Apps

Install the dependencies required by the existing housing-model notebooks:

```bash
pip install kagglehub pandas matplotlib scikit-learn statsmodels
```

Open a notebook in VS Code or Jupyter and run its cells in order. Dependencies and setup instructions for new case studies will be added with those projects.

### Running the Cloud ☁️ RAG Assistant

To launch the local documentation assistant in your browser:

```bash
cd modern_ml_reponsible_ai/rag_assistant
pip install -r requirements.txt
streamlit run app.py
```
