from flask import Flask, render_template, request
from models import get_qa_pipeline, get_sentence_model
from retrieval import (
    load_dataset,
    detect_domain,
    find_relevant_sections,
    answer_question_over_text,
    recommend_cases
)

app = Flask(__name__)

# ---------------- Global Caches ----------------
qa_pipeline = None
sentence_model = None
dataset = None


# ---------------- Model & Data Loading ----------------
def ensure_models_loaded():
    global qa_pipeline, sentence_model
    if qa_pipeline is None:
        qa_pipeline = get_qa_pipeline()
    if sentence_model is None:
        sentence_model = get_sentence_model()


def ensure_data_loaded():
    global dataset
    if dataset is None:
        dataset = load_dataset("data/legal_dataset.json")


# ---------------- Main Page ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ---------------- Core QA Logic ----------------
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("index.html", error="Please enter a question.")

    ensure_models_loaded()
    ensure_data_loaded()

    # ---------- Step 1: Domain Detection ----------
    domain_key = detect_domain(question, dataset["keywords"])
    domain_statutes = [s for s in dataset["statutes"] if s["domain"] == domain_key]
    domain_cases = [c for c in dataset["cases"] if c["domain"] == domain_key]

    # ✅ Fallbacks for robustness
    if not domain_statutes:
        domain_statutes = dataset["statutes"]
    if not domain_cases:
        domain_cases = dataset["cases"]

    # ---------- Step 2: Semantic Section Retrieval ----------
    relevant_sections = find_relevant_sections(question, domain_statutes, sentence_model, top_k=2)
    if not relevant_sections:
        return render_template(
            "result.html",
            question=question,
            answer_structured={
                "law": "No relevant section found",
                "description": "The system could not find any law matching your query.",
                "punishment": "N/A",
                "domain": "Unknown",
                "confidence": "N/A"
            },
            related_cases=[]
        )

    # ---------- Step 3: Extractive QA ----------
    best_answer, best_score, best_section = "No answer found.", 0.0, relevant_sections[0]
    for section in relevant_sections:
        answer, score = answer_question_over_text(question, section["summary"], qa_pipeline)
        if score > best_score:
            best_answer, best_score, best_section = answer, score, section

    # ---------- Step 4: Structured Answer Formatting ----------
    if len(best_answer.strip()) < 15 or "No direct" in best_answer:
        structured_answer = {
            "law": f"{best_section['title']} ({best_section['section']})",
            "description": best_section['summary'],
            "punishment": "As per this law, punishment may include imprisonment or fine, depending on severity.",
            "domain": best_section['domain'],
            "confidence": round(best_score, 2)
        }
    else:
        structured_answer = {
            "law": f"{best_section['title']} ({best_section['section']})",
            "description": f"{best_answer.strip().capitalize()}. {best_section['summary']}",
            "punishment": "This offence is punishable as defined under the respective section of law.",
            "domain": best_section['domain'],
            "confidence": round(best_score, 2)
        }

    # ---------- Step 5: Related Case Recommendations ----------
    related_cases = recommend_cases(question, domain_cases, sentence_model)

    # ---------- Step 6: Render Structured Output ----------
    return render_template(
        "result.html",
        question=question,
        answer_structured=structured_answer,
        related_cases=related_cases
    )


# ---------------- Run App ----------------
if __name__ == "__main__":
    ensure_data_loaded()
    app.run(host="0.0.0.0", port=5000, debug=True)
