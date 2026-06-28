from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import threading
from model.training import load_model

app = FastAPI(title="Sentiment Analysis")
lock = threading.Lock()

total_analyzed = 0
label_counts = {"pos": 0, "neg": 0, "neu": 0}
compound_sum = 0.0

NEU_THRESHOLD = 0.65

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

model = load_model()
classes = list(model.classes_)

def class_index(*labels) -> int:
    for label in labels:
        if label in classes:
            return classes.index(label)
    raise ValueError(f"Model is missing expected class labels: {labels}")

POS_CLASS_INDEX = class_index("pos", "positive", 1)
NEG_CLASS_INDEX = class_index("neg", "negative", 0)

def analyze_text(text: str) -> SentimentResult:
    probs = model.predict_proba([text])[0]
    pos_prob = float(probs[POS_CLASS_INDEX])
    neg_prob = float(probs[NEG_CLASS_INDEX])
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
