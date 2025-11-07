import os
from functools import lru_cache


def get_qa_pipeline():
    # Avoid importing torchvision for text-only pipelines to prevent optional dependency issues
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    from transformers import pipeline

    # DistilBERT extractive QA
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


@lru_cache(maxsize=1)
def get_sentence_model():
    from sentence_transformers import SentenceTransformer

    # Sentence-BERT for semantic similarity
    return SentenceTransformer("all-MiniLM-L6-v2")
