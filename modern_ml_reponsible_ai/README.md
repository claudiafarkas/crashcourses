# Topic: Modern ML & Responsible AI

Here I am learning the mechanics behind modern AI systems, how to evaluate them with evidence, and where their risks and limitations show up.

## Notebooks & Applications

- [RAG from Scratch](rag_from_scratch.ipynb): Document ingestion, chunking, embeddings, retrieval, generation, and failure modes.
- [Cloud ☁️ (ML Foundations RAG Assistant)](rag_assistant/app.py): A lightweight, editorial-styled Streamlit chatbot that indexes and queries the notebooks in `foundations_and_models/` as live documentation with grounded cell citations.
- [LLM Evaluation](llm_evaluation.ipynb): Evaluation sets, retrieval metrics, groundedness, relevance, latency, and cost.
- [Model Explainability](model_explainability.ipynb): Global and local explanations, permutation importance, partial dependence, and SHAP.
- [Responsible AI & Fairness](responsible_ai_fairness.ipynb): Bias, proxy variables, subgroup evaluation, privacy, and deployment safeguards.

The RAG case study applies the mechanics and evaluation practices introduced here to a specific use case.

## Running the Cloud ☁️ RAG Assistant Locally

To launch the local Streamlit chat assistant:

```bash
cd modern_ml_reponsible_ai/rag_assistant
pip install -r requirements.txt
streamlit run app.py
```
