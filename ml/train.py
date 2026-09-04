"""
Generates the pre-trained model artifacts (model.pkl, vectorizer.pkl)
used by both the FastAPI service and the Celery worker.

In real production systems, training happens completely separately from
serving. This script simulates that offline step with a small, hardcoded
sentiment-classification dataset.

Usage:
    python -m ml.train
"""
import logging
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"

# A small, hand-labeled dummy dataset for sentiment classification.
TEXTS = [
    "I love this product, it works perfectly",
    "This is amazing, best purchase ever",
    "Absolutely fantastic experience, highly recommend",
    "Great quality and fast shipping, very happy",
    "This is terrible, complete waste of money",
    "Worst customer service I have ever experienced",
    "Awful quality, broke after one day",
    "I hate this, do not buy it",
    "It is okay, nothing special about it",
    "Average product, does what it says",
    "Not bad, not great, just okay",
    "It works as expected, nothing more nothing less",
]

LABELS = [
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral",
]


def train_and_save() -> None:
    logger.info("Starting training with %d samples", len(TEXTS))

    cleaned_texts = [t.strip().lower() for t in TEXTS]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(cleaned_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(features, LABELS)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved vectorizer to %s", VECTORIZER_PATH)


if __name__ == "__main__":
    train_and_save()
