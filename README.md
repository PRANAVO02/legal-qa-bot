# AI-Powered Legal Question Answering Bot

A Flask web app that answers Indian law questions using extractive QA (DistilBERT) and recommends related cases via semantic similarity (Sentence-BERT).

## Features
- Extract answers from a small IPC corpus using `distilbert-base-cased-distilled-squad`.
- Recommend similar case summaries using `all-MiniLM-L6-v2` SentenceTransformer.
- Simple Bootstrap UI.

## Project Structure
```
legal-qa-bot/
├─ app.py
├─ models.py
├─ retrieval.py
├─ requirements.txt
├─ data/
│  ├─ legal_corpus.json
│  └─ case_summaries.json
├─ templates/
│  ├─ index.html
│  └─ result.html
└─ static/
   └─ css/
      └─ styles.css
```

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `torch` fails, install a compatible wheel from https://pytorch.org/get-started/locally/ and then re-run the above.

## Run
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Notes
- On first run, models will download from Hugging Face. This requires internet access.
- For larger datasets, consider caching embeddings or using a vector DB.
- This sample corpus is minimal; replace `data/*.json` with richer datasets.
