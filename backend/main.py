from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import csv
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

app = FastAPI(title="Sentiment Analysis")
lock = threading.Lock()

total_analyzed = 0
label_counts = {"pos": 0, "neg": 0, "neu": 0}
compound_sum = 0.0

NEU_THRESHOLD = 0.6

class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[list[str]] = None

class SentimentResult(BaseModel):
    text: str
    compound: float
    pos: float
    neg: float
    neu: float
    label: str

class AnalyzeResponse(BaseModel):
    results: list[SentimentResult]

def _build_model():
    csv_path = Path(__file__).resolve().parent / "training_data.csv"
    texts, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["sentiment"])

   
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    eval_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Dataset : {len(texts)} total samples")
    print(f"Train   : {len(X_train)} | Test : {len(X_test)}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nClassification Report (held-out test set):")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"{'='*50}\n")

   
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    model.fit(texts, labels)
    return model

model = _build_model()
classes = list(model.classes_)

def analyze_text(text: str) -> SentimentResult:
    probs = model.predict_proba([text])[0]
    pos_prob = float(probs[classes.index("pos")])
    neg_prob = float(probs[classes.index("neg")])
    conf = max(pos_prob, neg_prob)
    if conf < NEU_THRESHOLD:
        label = "neu"
    else:
        label = "pos" if pos_prob > neg_prob else "neg"
    compound = round(pos_prob - neg_prob, 4)
    neu_score = round(1 - abs(compound), 4)
    return SentimentResult(
        text=text, compound=compound,
        pos=round(pos_prob, 4), neg=round(neg_prob, 4),
        neu=neu_score, label=label,
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "sklearn",
        "version": "1.0"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    texts = []
    if req.texts:
        texts = req.texts
    elif req.text:
        texts = [req.text]
    results = [analyze_text(t) for t in texts]
    with lock:
        global total_analyzed, compound_sum
        for r in results:
            total_analyzed += 1
            label_counts[r.label] += 1
            compound_sum += r.compound
    return AnalyzeResponse(results=results)

@app.get("/metrics")
def metrics():
    with lock:
        avg = round(compound_sum / total_analyzed, 4) if total_analyzed else 0.0
        return {
            "total_analyzed": total_analyzed,
            "label_counts": dict(label_counts),
            "avg_compound": avg
        }

frontend_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
