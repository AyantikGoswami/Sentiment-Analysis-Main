# Sentiment Analysis System

Type text, get a positive/negative/neutral verdict. 

# Features
- Real-time sentiment analysis
- Positive, negative, and neutral classification
- Confidence-based prediction handling
- REST API built with FastAPI
- Modern web interface with dynamic themes
- Batch analysis support

## Get the Code

```bash
git clone https://github.com/SoumyabrataBhat/Sentiment-Analysis
cd Sentiment-Analysis
```

## Quick Start

```bash
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..
python backend/main.py
# → http://127.0.0.1:8000
```

## How It Works

Trains a **TF-IDF + Logistic Regression** model on 3,000 labelled sentences from Amazon, IMDb, and Yelp reviews (UCI dataset). For a detailed mathematical and structural breakdown of the pipeline, read the [Technical Report (PDF)](./docs/sentiment-analysis-report.pdf).

- Converts text to numeric vectors using TF-IDF (word importance scores)
- Logistic Regression learns which words predict positive or negative
- Confident predictions are labelled pos/neg; uncertain ones (<60% confidence) are labelled neutral
- Compound score = probability(positive) − probability(negative), ranges from −1 to +1

```
Input → TF-IDF vectorize → Logistic Regression predict_proba → Confidence gate → Label
```

## Backend API

Single FastAPI file (`backend/main.py`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{status, engine}` |
| `/analyze` | POST | `{"text": "..."}` or `{"texts": ["..."]}` — returns sentiment results |
| `/metrics` | GET | Cumulative stats: total analyzed, label distribution, avg compound |

Serves the built frontend from `frontend/dist/` as static files.

## Screenshots

<div align="center">
  <h3>1. Positive Sentiment (Dynamic Green Theme)</h3>
  <img src="screenshots/positive.png" alt="Positive Sentiment Analysis" width="600" />
  <p><i>Analysis of a positive statement ("kolkata is such a nice place") displaying positive polarity scores, compound calculation, and the active green background theme.</i></p>
</div>

<br/>

<div align="center">
  <h3>2. Negative Sentiment (Dynamic Burgundy Theme)</h3>
  <img src="screenshots/negative.png" alt="Negative Sentiment Analysis" width="600" />
  <p><i>Detection of a negative statement ("that movie is not for me") illustrating the negative score distribution, negative compound index, and transition to the burgundy background theme.</i></p>
</div>

<br/>

<div align="center">
  <h3>3. Neutral Sentiment (Dynamic Blue Theme)</h3>
  <img src="screenshots/neutral.png" alt="Neutral Sentiment Analysis" width="600" />
  <p><i>Evaluation of a neutral statement ("i want to eat biriyani") demonstrating the neutral classification and transition to the dark blue background theme.</i></p>
</div>

