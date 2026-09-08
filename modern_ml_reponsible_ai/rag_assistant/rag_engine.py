"""
RAG Engine module for querying ML Foundations Notebooks.

You will implement the retrieval, embedding, and generation components here 
(or port them directly from your rag_from_scratch.ipynb experiments).
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
    "supervised learning": {"regression", "classification", "ridge", "lasso", "logistic", "random forest", "decision tree", "prediction"},
    "unsupervised learning": {"clustering", "cluster", "k-means", "dbscan", "pca", "anomaly", "unsupervised", "dimensionality"},
}


def classify_question(question: str) -> Dict[str, Any]:
    """Return an inspectable topic and question type for the UI scaffold."""
    normalized_question = question.lower()
    topic_scores = {
        topic: sum(keyword in normalized_question for keyword in keywords)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    topic = max(topic_scores, key=topic_scores.get)
    if topic_scores[topic] == 0:
        topic = "general ML"

    if any(marker in normalized_question for marker in {"break down", "simpler", "clarify", "more"}):
        question_type = "clarification"
    elif any(marker in normalized_question for marker in {"how", "show", "code", "implement"}):
        question_type = "implementation"
    elif any(marker in normalized_question for marker in {"why", "explain", "mean", "define", "what is"}):
        question_type = "concept explanation"
    elif any(marker in normalized_question for marker in {"compare", "versus", "difference", "better"}):
        question_type = "comparison"
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
        """
        Parse notebooks, chunk cells (markdown + code), generate embeddings,
        and store them in the vector database.
        
        Returns:
            int: Number of indexed chunks.
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
        """
        Process a user query through the RAG pipeline:
        1. Embed the query.
        2. Retrieve top-k relevant chunks from vector store.
        3. Construct prompt with context.
        4. Generate grounded response with citations.
        
        Args:
            user_query: The natural language question.
            top_k: Number of retrieved chunks to provide as context.
            
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
        sources = []
        for index in np.argsort(similarities)[::-1]:
            if similarities[index] <= 0:
                continue
            document = self.documents[index]
            sources.append({
                **document,
                "score": round(float(similarities[index]), 4),
            })
            if len(sources) >= top_k:
                break

        formatted_sources = "\n\n".join(
            f"Source {index}: {source['notebook']} ({source['chunk_type']}, score={source['score']})\n"
            f"{source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        prompt = f"""You are Cloud, an assistant for the ML Foundations notebooks.

Answer using only the retrieved notebook evidence below. Use recent conversation only to understand follow-up references. Explain clearly, do not invent unsupported facts, and say when the sources are insufficient. Mention the source notebook for factual claims.

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
        }
