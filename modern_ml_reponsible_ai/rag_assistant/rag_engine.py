"""Application version of the RAG pipeline explored in ``rag_sandbox.ipynb``.

The sandbox exposes each stage for learning and debugging. This module keeps
the same stages behind methods that Streamlit can call:

1. ``index_notebooks`` parses and chunks the source notebooks, then builds the
    TF-IDF retrieval index.
2. ``query`` classifies the question, retrieves evidence, builds a grounded
    prompt, and asks Ollama for the response.
"""

import glob
import os
from typing import List, Dict, Any

import nbformat
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


TOPIC_KEYWORDS = {
    "preprocessing": {"preprocessing", "missing", "imputation", "encoding", "categorical", "scaling", "leakage"},
    "model evaluation": {"evaluation", "metric", "metrics", "mae", "rmse", "roc", "auc", "precision", "recall", "calibration", "threshold", "error"},
    "supervised learning": {"supervised", "regression", "classification", "ridge", "lasso", "logistic", "random forest", "decision tree", "prediction"},
    "unsupervised learning": {"clustering", "cluster", "k-means", "dbscan", "pca", "anomaly", "unsupervised", "dimensionality"},
}


def classify_question(question: str) -> Dict[str, Any]:
    """Return an inspectable topic and question type for the UI scaffold."""
    normalized_question = question.lower()
    topic_scores = {
        topic: sum(keyword in normalized_question for keyword in keywords)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    matched_topics = [
        topic for topic, score in topic_scores.items()
        if score > 0
    ]
    if {
        "supervised learning",
        "unsupervised learning",
    }.issubset(matched_topics):
        topic = "supervised vs unsupervised learning"
    elif matched_topics:
        topic = max(matched_topics, key=lambda item: topic_scores[item])
    else:
        topic = "general ML"

    if any(marker in normalized_question for marker in {"break down", "simpler", "clarify", "more"}):
        question_type = "clarification"
    elif any(marker in normalized_question for marker in {"compare", "versus", "difference", "better"}):
        question_type = "comparison"
    elif "how different" in normalized_question:
        question_type = "comparison"
    elif any(marker in normalized_question for marker in {"how", "show", "code", "implement"}):
        question_type = "implementation"
    elif any(marker in normalized_question for marker in {"why", "explain", "mean", "define", "what is"}):
        question_type = "concept explanation"
    else:
        question_type = "general question"

    return {
        "topic": topic,
        "question_type": question_type,
        "topic_scores": topic_scores,
    }


def previous_topic(conversation: List[Dict[str, str]]) -> str:
    """Return the most recent specific topic from conversation history."""
    for message in reversed(conversation):
        topic = message.get("topic")
        if topic and topic != "general ML":
            return topic
    return "general ML"


class RAGEngine:
    def __init__(self, notebooks_dir: str = None):
        """
        Initialize the RAG Engine. (what does this do? It sets up the directory for notebooks, initializes the indexing flag, and prepares the document storage for later retrieval and embedding.)
        
        Args:
            notebooks_dir: Path to the directory containing the target notebooks.
                           Defaults to the foundations_and_models directory.
        """
        if notebooks_dir is None:
            # Point to foundations_and_models relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.notebooks_dir = os.path.join(base_dir, "foundations_and_models")
        else:
            self.notebooks_dir = notebooks_dir

        # These are the notebook variables from the sandbox, stored on the
        # engine instance so Streamlit can reuse them across reruns.
        self.indexed = False
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.chunk_vectors = None

    def get_notebook_list(self) -> List[str]:
        """List available .ipynb notebooks in the target directory."""
        if not os.path.exists(self.notebooks_dir):
            return []
        return [
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.notebooks_dir, "*.ipynb"))
        ]

    def index_notebooks(self) -> int:
        """Build the in-memory index from the Foundations notebooks.

        Sandbox mapping:
        - Phase 1 extraction becomes ``notebook_cells`` here.
        - Phase 2 concept and paired chunks become ``self.documents``.
        - Phase 3 TF-IDF vectorization becomes ``self.vectorizer`` and
          ``self.chunk_vectors``.

        Returns:
            int: Number of indexed concept and implementation chunks.
        """
        notebook_cells = []
        for file_path in sorted(glob.glob(os.path.join(self.notebooks_dir, "*.ipynb"))):
            file_name = os.path.basename(file_path)
            try:
                notebook = nbformat.read(file_path, as_version=4)
            except Exception:
                continue

            for cell_index, cell in enumerate(notebook.cells):
                if cell.cell_type not in {"markdown", "code"}:
                    continue
                content = cell.source.strip()
                if content:
                    notebook_cells.append({
                        "content": content,
                        "notebook": file_name,
                        "cell_index": cell_index,
                        "cell_type": cell.cell_type,
                    })

        concept_chunks = []
        implementation_chunks = []
        notebook_names = sorted({record["notebook"] for record in notebook_cells})

        for notebook_name in notebook_names:
            records = sorted(
                [record for record in notebook_cells if record["notebook"] == notebook_name],
                key=lambda record: record["cell_index"],
            )
            latest_markdown = None
            for record in records:
                if record["cell_type"] == "markdown":
                    latest_markdown = record
                    concept_chunks.append({
                        **record,
                        "chunk_type": "concept",
                    })
                    continue

                explanation = latest_markdown["content"] if latest_markdown else ""
                implementation_chunks.append({
                    "content": (
                        f"Notebook: {notebook_name}\n"
                        f"Explanation:\n{explanation}\n\n"
                        f"Code:\n{record['content']}"
                    ),
                    "notebook": notebook_name,
                    "markdown_cell_index": latest_markdown["cell_index"] if latest_markdown else None,
                    "code_cell_index": record["cell_index"],
                    "cell_type": "markdown_and_code",
                    "chunk_type": "implementation",
                })

        # The sandbox keeps these as separate lists for inspection. The app
        # combines them into one searchable collection.
        self.documents = concept_chunks + implementation_chunks
        if not self.documents:
            raise ValueError(f"No Markdown or code cells found in {self.notebooks_dir}.")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(
            [document["content"] for document in self.documents]
        )
        self.indexed = True
        return len(self.documents)

    def query(
        self,
        user_query: str,
        top_k: int = 3,
        conversation: List[Dict[str, str]] = None,
        model: str = "qwen2.5:7b",
    ) -> Dict[str, Any]:
        """Run the application query path.

        Sandbox mapping:
        - Phase 4 topic and follow-up logic becomes the classification and
          conversation handling below.
        - Phase 3 ``search_chunks`` becomes the TF-IDF/cosine similarity loop.
        - Phase 5 ``build_grounded_prompt`` becomes the prompt string below.
        - Phase 5 Ollama call becomes the generated answer returned to the UI.

        Args:
            user_query: The natural language question.
            top_k: Number of retrieved chunks to provide as context.
            conversation: Recent Streamlit messages for follow-up questions.
            model: Ollama model name.
            
        Returns:
            dict with keys:
                - 'answer': str (LLM response)
                - 'sources': list of dicts [{'notebook': str, 'cell_type': str, 'content': str, 'score': float}]
        """
        if not self.indexed:
            self.index_notebooks()

        conversation = conversation or []
        classification = classify_question(user_query)
        topic = classification["topic"]
        if topic == "general ML" and classification["question_type"] == "clarification":
            topic = previous_topic(conversation)
        recent_context = "\n".join(
            message.get("content", "") for message in conversation[-4:]
        )
        retrieval_query = user_query
        if recent_context and classification["question_type"] == "clarification":
            retrieval_query = f"Topic: {topic}\nPrevious conversation:\n{recent_context}\nQuestion:\n{user_query}"

        query_vector = self.vectorizer.transform([retrieval_query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors).ravel()
        ranked_indices = np.argsort(similarities)[::-1]

        # TF-IDF can legitimately return zero similarity for broad or vocabulary-miss
        # queries. In that case, keep the nearest chunks instead of dropping all
        # evidence silently so the UI still shows citations and the prompt stays grounded.
        positive_indices = [index for index in ranked_indices if similarities[index] > 0]
        fallback_indices = [index for index in ranked_indices if similarities[index] <= 0]
        selected_indices = positive_indices[:top_k]
        if len(selected_indices) < top_k:
            remaining_slots = top_k - len(selected_indices)
            selected_indices.extend(fallback_indices[:remaining_slots])

        sources = []
        for index in selected_indices:
            document = self.documents[index]
            sources.append({
                **document,
                "score": round(float(similarities[index]), 4),
            })

        max_similarity = max((float(doc.get("score", 0.0)) for doc in sources), default=0.0) if sources else 0.0
        retrieval_status = "grounded" if max_similarity >= 0.10 else "fallback"

        if retrieval_status == "fallback":
            sources = []
            formatted_sources = "" 
            guidance = """No notebook evidence was retrieved for this question. Answer cautiously and clearly state that the response is based on general model knowledge rather than the notebook corpus. Do not pretend the answer is directly sourced from the notebooks when no evidence was found."""
        else:
            formatted_sources = "\n\n".join(
                f"Source {index}: {source['notebook']} ({source['chunk_type']}, score={source['score']})\n"
                f"{source['content']}"
                for index, source in enumerate(sources, start=1)
            )
            guidance = """Answer using only the retrieved notebook evidence below. Use recent conversation only to understand follow-up references. Explain clearly, do not invent unsupported facts, and say when the sources are insufficient. Mention the source notebook for factual claims."""

        prompt = f"""You are Cloud, an assistant for the ML Foundations notebooks.

{guidance}

Topic: {topic}
Question type: {classification['question_type']}
Recent conversation:
{recent_context}

User question:
{user_query}

Retrieved notebook evidence:
{formatted_sources}
"""

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "answer": response["message"]["content"],
            "sources": sources,
            "topic": topic,
            "question_type": classification["question_type"],
            "topic_scores": classification["topic_scores"],
            "retrieval_status": retrieval_status,
        }
