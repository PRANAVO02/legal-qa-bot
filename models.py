import os
import sys
from functools import lru_cache

# 🚫 Block optional heavy deps we don't need (vision, tf/keras) BEFORE any transformers imports
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("tensorflow", None)
sys.modules.setdefault("tf_keras", None)
sys.modules.setdefault("keras", None)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def get_qa_pipeline():
    """
    Returns a Hugging Face extractive QA pipeline using DistilBERT.
    Safe version that prevents torchvision import issues.
    """
    # Avoid importing torchvision for text-only pipelines to prevent optional dependency issues
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    # Force Transformers to avoid TensorFlow/keras imports (redundant safeguard)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    from transformers import pipeline

    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        framework="pt",
    )


@lru_cache(maxsize=1)
def get_sentence_model():
    """Loads a lightweight SentenceTransformer model (PyTorch-only)."""
    # Ensure TF is still disabled before import
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    sys.modules.setdefault("tensorflow", None)
    sys.modules.setdefault("tf_keras", None)
    sys.modules.setdefault("keras", None)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
