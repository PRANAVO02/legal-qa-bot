#retrival.py
import json
import numpy as np
from typing import List, Dict, Any
import re

# ---------- Load Dataset ----------
def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Domain Detection (Weighted & Smarter) ----------
def detect_domain(question: str, keywords: Dict[str, List[str]]) -> str:
    """Detect domain based on keyword match frequency with weighted scoring."""
    q_lower = question.lower()
    best_match = ("GENERAL", 0)
    for domain, keys in keywords.items():
        score = sum(2 if len(k.split()) > 1 and k in q_lower else 1 for k in keys if k in q_lower)
        if score > best_match[1]:
            best_match = (domain, score)
    return best_match[0]

# ---------- Semantic Section Finder ----------
def find_relevant_sections(question: str, statutes: List[Dict[str, Any]], model, top_k=2):
    """Find top_k most relevant legal sections using semantic similarity."""
    if not statutes:
        return []

    texts = [s.get("summary", "").strip() for s in statutes if s.get("summary", "").strip()]
    if not texts:
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
        return res.get("answer", ""), float(res.get("score", 0.0))
    except Exception:
        return "No direct extract found.", 0.0


# ---------- Explicit Section Resolver ----------
def find_sections_by_hint(question: str, statutes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If the question explicitly mentions a section number like 'IPC 420', prioritize those statutes."""
    q = question.lower()
    # Extract 2-4 digit numbers that could be section numbers
    numbers = set(re.findall(r"\b\d{2,4}\b", q))
    if not numbers:
        return []
    matches = []
    for s in statutes:
        title = str(s.get("title", "")).lower()
        section = str(s.get("section", "")).lower()
        sid = str(s.get("id", "")).lower()
        if any(n in title or n in section or n in sid for n in numbers):
            matches.append(s)
    return matches

# ---------- Case Recommendation ----------
def recommend_cases(question: str, cases: List[Dict[str, Any]], model, top_k=3):
    """Find top_k similar landmark cases using semantic similarity."""
    if not cases:
        return []
    query_emb = model.encode([question], convert_to_numpy=True)[0]
    case_texts = [c.get("snippet", "") for c in cases]
    case_embs = model.encode(case_texts, convert_to_numpy=True)

    # Cosine similarity with small eps to avoid division by zero
    qn = np.linalg.norm(query_emb) + 1e-12
    cn = np.linalg.norm(case_embs, axis=1) + 1e-12
    sims = (case_embs @ query_emb) / (cn * qn)

    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        case_name = cases[i]["title"]
        url = cases[i].get("url", "")
        results.append({
            "case": case_name,
            "snippet": cases[i].get("snippet", ""),
            "similarity": float(sims[i]),
            "url": url,
        })
    return results
