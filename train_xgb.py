"""Retrain the project's existing XGBoost classifier with the added reviews."""

import pickle
import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Models"


def clean_review(text, blocked_words, stemmer):
    words = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    return " ".join(stemmer.stem(word) for word in words if word not in blocked_words)


def main():
    alexa = pd.read_csv(ROOT / "Data" / "amazon_alexa.tsv", sep="\t")
    alexa = alexa.dropna(subset=["verified_reviews", "feedback"])
    supplemental = pd.read_csv(ROOT / "Data" / "amazon_negative_reviews.csv")
    data = pd.concat(
        [alexa[["verified_reviews", "feedback"]], supplemental],
        ignore_index=True,
    )
    data["feedback"] = data["feedback"].astype(int)

    blocked_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    texts = [
        clean_review(text, blocked_words, stemmer)
        for text in data["verified_reviews"].tolist()
    ]
    labels = data["feedback"].to_numpy()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=15,
        stratify=labels,
    )

    evaluation_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    train_features = evaluation_vectorizer.fit_transform(train_texts).toarray()
    test_features = evaluation_vectorizer.transform(test_texts).toarray()
    evaluation_scaler = MinMaxScaler()
    train_features = evaluation_scaler.fit_transform(train_features)
    test_features = evaluation_scaler.transform(test_features)

    evaluation_model = XGBClassifier(random_state=15)
    sample_weights = compute_sample_weight("balanced", train_labels)
    evaluation_model.fit(train_features, train_labels, sample_weight=sample_weights)
    test_predictions = evaluation_model.predict(test_features)
    print("Holdout confusion matrix [negative=0, positive=1]:")
    print(confusion_matrix(test_labels, test_predictions, labels=[0, 1]))
    print("Holdout classification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    # Fit deployment artifacts on all available training rows.
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts).toarray()
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    model = XGBClassifier(random_state=15)
    model.fit(features, labels, sample_weight=compute_sample_weight("balanced", labels))

    example = "The device stopped working after two days."
    example_features = scaler.transform(
        vectorizer.transform([clean_review(example, blocked_words, stemmer)]).toarray()
    )
    example_probabilities = model.predict_proba(example_features)[0]
    example_label = int(model.classes_[example_probabilities.argmax()])
    print(
        "Deployment example prediction:",
        "Positive" if example_label == 1 else "Negative",
        "(negative probability:", round(float(example_probabilities[0]), 3), ")",
    )

    with (MODEL_DIR / "model_xgb.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (MODEL_DIR / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)
    with (MODEL_DIR / "countVectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)
    print("Saved updated XGBoost artifacts using", len(data), "training rows.")


if __name__ == "__main__":
    main()
