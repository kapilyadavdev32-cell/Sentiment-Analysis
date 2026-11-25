# Amazon Alexa Review Sentiment Analysis

This project trains a binary sentiment classifier from Amazon Alexa reviews and provides single text and CSV/TSV prediction interfaces. The Flask API uses the bundled XGBoost model and preprocessing artifacts in `Models/`; the HTML interface is served from the Flask root route, and `main.py` provides an optional Streamlit client.

Walkthrough video: [End to End NLP Project | Sentiment Analysis on Amazon Reviews](https://www.youtube.com/watch?v=6A2w-KYG4Ko)

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies and download the NLTK stopword list:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Run the Flask application

From the repository root:

```bash
python api.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The API health check is at `/test`.

## Run the Streamlit client

Keep the Flask application running, open a second terminal in the repository root, activate the same environment, then run:

```bash
streamlit run main.py
```

The client opens at the local Streamlit URL printed in the terminal.

## Prediction inputs

- Single prediction: enter review text in either interface.
- Bulk prediction: upload a comma-separated CSV or tab-separated TSV containing a `Sentence` or `verified_reviews` column. The response is a CSV with a `Predicted sentiment` column and the Flask response includes a sentiment distribution chart. Blank reviews are marked `Unclassified`.

The model predicts positive or negative sentiment. The dataset is strongly imbalanced, so accuracy alone is not a sufficient measure of model quality.

## Training data supplement

`Data/amazon_negative_reviews.csv` contains 1,908 negative reviews from the MIT-licensed [Cleanlab Amazon Reviews dataset](https://huggingface.co/datasets/Cleanlab/amazon-reviews). The notebook appends them to the Alexa training rows before text vectorization. To retrain the app's XGBoost artifacts reproducibly, run `python train_xgb.py`, then restart Flask. The script evaluates on a stratified holdout set and refits the deployment model on all training rows. Details and label mapping are in [Data/README.md](Data/README.md).
