import json
import numpy as np
from typing import List, Dict, Any

# ---------- Load Dataset ----------
def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Domain Detection ----------
def detect_domain(question: str, keywords: Dict[str, List[str]]) -> str:
    q_lower = question.lower()
    best_match = ("GENERAL", 0)
    for domain, keys in keywords.items():
        match_count = sum(k in q_lower for k in keys)
        if match_count > best_match[1]:
            best_match = (domain, match_count)
    return best_match[0]

# ---------- Semantic Section Finder ----------
def find_relevant_sections(question: str, statutes: List[Dict[str, Any]], model, top_k=2):
    """Find top_k most relevant legal sections using semantic similarity."""
    if not statutes:  # Prevent empty input
        return []

    texts = [s.get("summary", "") for s in statutes if s.get("summary")]
    if not texts:  # Double-check empty texts
        return []

    query_emb = model.encode([question], convert_to_numpy=True)[0]
    text_embs = model.encode(texts, convert_to_numpy=True)

    sims = (text_embs @ query_emb) / (
        np.linalg.norm(text_embs, axis=1) * np.linalg.norm(query_emb)
    )

    top_idxs = np.argsort(-sims)[: min(top_k, len(sims))]
    return [statutes[i] for i in top_idxs]


# ---------- Question Answering ----------
def answer_question_over_text(question: str, context: str, qa_pipeline):
    try:
        res = qa_pipeline(question=question, context=context)
        return res.get("answer"), float(res.get("score", 0))
    except Exception:
        return "No direct extract found.", 0.0

# ---------- Case Recommendation ----------
def recommend_cases(question: str, cases: List[Dict[str, Any]], model, top_k=3):
    if not cases:
        return []
    query_emb = model.encode([question], convert_to_numpy=True)[0]
    case_embs = model.encode([c["snippet"] for c in cases], convert_to_numpy=True)

    sims = (case_embs @ query_emb) / (np.linalg.norm(case_embs, axis=1) * np.linalg.norm(query_emb))
    idxs = np.argsort(-sims)[:top_k]

    return [
        {
            "case": cases[i]["title"],
            "snippet": cases[i]["snippet"],
            "similarity": float(sims[i])
        }
        for i in idxs
    ]
