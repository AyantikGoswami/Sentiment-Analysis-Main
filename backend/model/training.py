import csv
import html
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent / "IMDB.csv"
MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pkl"
LABEL_MAP = {
    "positive": "pos",
    "negative": "neg",
}


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label: str) -> str:
    normalized = LABEL_MAP.get(str(label).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported sentiment label: {label!r}")
    return normalized


def load_training_data(csv_path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text") or row.get("review")
            label = row.get("sentiment")
            if not text or not label:
                continue

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            texts.append(cleaned_text)
            labels.append(normalize_label(label))
    return texts, labels


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,       
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,                 
            sublinear_tf=True,         
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",  
        )),
    ])


def build_model(csv_path: Path = DATA_PATH, show_report: bool = True) -> Pipeline:
    texts, labels = load_training_data(csv_path)

    if show_report:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        eval_model = create_pipeline()
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)

        print(f"\n{'=' * 50}")
        print(f"Dataset : {len(texts)} total samples")
        print(f"Train   : {len(X_train)} | Test : {len(X_test)}")
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
        print("\nClassification Report (held-out test set):")
        print(classification_report(y_test, y_pred, digits=3))
        print(f"{'=' * 50}\n")

    model = create_pipeline()
    model.fit(texts, labels)
    return model


def train_and_save_model(
    model_path: Path = MODEL_PATH,
    csv_path: Path = DATA_PATH,
    show_report: bool = True,
) -> Pipeline:
    model = build_model(csv_path=csv_path, show_report=show_report)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    if model_path.exists():
        return joblib.load(model_path)
    return train_and_save_model(model_path=model_path)


if __name__ == "__main__":
    train_and_save_model()
    print(f"Model saved to: {MODEL_PATH}")