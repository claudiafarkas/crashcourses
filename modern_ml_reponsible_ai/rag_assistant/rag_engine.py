"""
RAG Engine module for querying ML Foundations Notebooks.

You will implement the retrieval, embedding, and generation components here 
(or port them directly from your rag_from_scratch.ipynb experiments).
"""

import os
import glob
from typing import List, Dict, Any


class RAGEngine:
    def __init__(self, notebooks_dir: str = None):
        """
        Initialize the RAG Engine.
        
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
        
        # TODO: Initialize your embedding model, vector store (e.g. Chroma/FAISS), and LLM client here

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
        # =========================================================================
        # TODO: Step 1 - Parse notebooks using `nbformat` or `json`.
        # TODO: Step 2 - Extract text & code cells with metadata (notebook, cell_idx).
        # TODO: Step 3 - Create vector embeddings and add to index.
        # =========================================================================
        self.indexed = True
        return len(self.get_notebook_list())

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
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
        # =========================================================================
        # TODO: Replace this placeholder response with your actual RAG retrieval & LLM call
        # =========================================================================
        
        # Placeholder response demonstrating the output format expected by app.py:
        mock_sources = [
            {
                "notebook": "model_evaluation_error_analysis.ipynb",
                "cell_type": "markdown",
                "score": 0.91,
                "content": "### Calibration and Decision Thresholds\nIn imbalanced classification, default 0.5 threshold can lead to low recall. Calibration curves evaluate predicted probabilities vs true frequencies."
            },
            {
                "notebook": "preprocessing.ipynb",
                "cell_type": "code",
                "score": 0.84,
                "content": "from sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder"
            }
        ]
        
        mock_answer = (
            f"*(RAG Engine Scaffold Mode)*\n\n"
            f"You asked: **\"{user_query}\"**\n\n"
            f"Once you connect your embedding model and LLM in `rag_engine.py` (or from `rag_from_scratch.ipynb`), "
            f"this will generate a grounded answer citing specific notebook cells from `{os.path.basename(self.notebooks_dir)}`."
        )
        
        return {
            "answer": mock_answer,
            "sources": mock_sources
        }
