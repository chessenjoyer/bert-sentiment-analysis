"""FastAPI demo for the sentiment model trained in HW_transformers.ipynb."""

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOCAL_MODEL = Path(__file__).parent / "fine_tuned_model"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = os.getenv(
    "MODEL_PATH", str(LOCAL_MODEL if LOCAL_MODEL.exists() else FALLBACK_MODEL)
)

app = FastAPI(
    title="BERT Sentiment Analysis",
    description="Binary sentiment classification for English movie reviews.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


tokenizer = None
model = None


def load_model() -> None:
    """Load once on first request so importing the API stays lightweight."""
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        load_model()
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

        # The notebook trains on IMDB: 0 = negative, 1 = positive.
        labels = {0: "Negative", 1: "Positive"}
        return {
            "prediction": labels.get(predicted_id, str(predicted_id)),
            "probabilities": {
                labels.get(i, str(i)): float(value)
                for i, value in enumerate(probabilities)
            },
        }
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Model is unavailable: {exc}") from exc
