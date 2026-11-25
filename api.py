import base64
import pickle
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the bundled model artifacts once, using paths independent of cwd."""
    with (MODEL_DIR / "model_xgb.pkl").open("rb") as model_file:
        predictor = pickle.load(model_file)
    with (MODEL_DIR / "scaler.pkl").open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with (MODEL_DIR / "countVectorizer.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return predictor, scaler, vectorizer


@lru_cache(maxsize=1)
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords are missing. Run: python -m nltk.downloader stopwords"
        ) from exc


def preprocess_text(text):
    words = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    blocked_words = get_stopwords()
    stemmer = PorterStemmer()
    return " ".join(
        stemmer.stem(word) for word in words if word not in blocked_words
    )


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        predictor, scaler, vectorizer = load_artifacts()
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file.filename:
                return jsonify({"error": "Choose a CSV or TSV file to upload."}), 400
            delimiter = (
                "\t"
                if uploaded_file.filename.lower().endswith((".tsv", ".tab"))
                else ","
            )
            try:
                data = pd.read_csv(uploaded_file, sep=delimiter)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                return jsonify({"error": f"Could not read the uploaded file: {exc}"}), 400

            text_column = next(
                (name for name in ("Sentence", "verified_reviews") if name in data.columns),
                None,
            )
            if text_column is None:
                return jsonify(
                    {
                        "error": "CSV/TSV must contain a 'Sentence' or 'verified_reviews' column."
                    }
                ), 400
            if data.empty:
                return jsonify({"error": "CSV/TSV must contain at least one row."}), 400
            if not data[text_column].fillna("").astype(str).str.strip().any():
                return jsonify({"error": f"'{text_column}' contains no review text."}), 400

            predictions, graph = bulk_prediction(
                predictor, scaler, vectorizer, data, text_column
            )
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getvalue()
            ).decode("ascii")
            return response

        payload = request.get_json(silent=True) or request.form
        text_input = payload.get("text")
        if not isinstance(text_input, str) or not text_input.strip():
            return jsonify({"error": "Provide non-empty text to classify."}), 400

        sentiment = single_prediction(
            predictor, scaler, vectorizer, text_input.strip()
        )
        return jsonify({"prediction": sentiment})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        app.logger.exception("Could not load model artifacts")
        return jsonify({"error": f"Could not load model artifacts: {exc}"}), 500
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the input and server logs."}), 500


def predict_labels(predictor, scaler, vectorizer, texts):
    corpus = [preprocess_text(text if isinstance(text, str) else "") for text in texts]
    labels = ["Unclassified"] * len(corpus)
    valid_rows = [index for index, text in enumerate(corpus) if text]
    if not valid_rows:
        return labels
    features = vectorizer.transform([corpus[index] for index in valid_rows]).toarray()
    scaled_features = scaler.transform(features)
    class_ids = predictor.predict_proba(scaled_features).argmax(axis=1)
    for index, class_id in zip(valid_rows, class_ids):
        labels[index] = sentiment_mapping(class_id)
    return labels


def single_prediction(predictor, scaler, cv, text_input):
    return predict_labels(predictor, scaler, cv, [text_input])[0]


def bulk_prediction(predictor, scaler, cv, data, text_column="Sentence"):
    result = data.copy()
    result["Predicted sentiment"] = predict_labels(
        predictor, scaler, cv, result[text_column].tolist()
    )
    predictions_csv = BytesIO()
    result.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    return predictions_csv, get_distribution_graph(result)


def get_distribution_graph(data):
    counts = data["Predicted sentiment"].value_counts().reindex(
        ["Positive", "Negative"], fill_value=0
    )
    visible_counts = counts[counts > 0]
    colors = {"Positive": "green", "Negative": "red"}
    fig, ax = plt.subplots(figsize=(5, 5))
    visible_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=[colors[label] for label in visible_counts.index],
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "black"},
        explode=[0.01] * len(visible_counts),
        title="Sentiment Distribution",
        ylabel="",
        ax=ax,
    )
    graph = BytesIO()
    fig.savefig(graph, format="png", bbox_inches="tight")
    plt.close(fig)
    graph.seek(0)
    return graph


def sentiment_mapping(class_id):
    return "Positive" if class_id == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
