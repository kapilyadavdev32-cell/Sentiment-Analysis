# Training data supplements

The source file uses `review_text` and `label`. This project file keeps only rows labeled `negative` and maps them to the notebook schema: `verified_reviews` is the text and `feedback` is `0` (negative). It contains 1,908 rows.

The notebook appends these rows to the Alexa reviews after its initial missing-row cleanup and before text vectorization. This changes the data used to train the existing XGBoost classifier; it does not change the classifier type. Run `python train_xgb.py` to evaluate and regenerate the app artifacts with the supplemented data.

`mixed_sentiment_test.csv` is a balanced 16-row smoke-test file for the upload interface. It includes an `Expected sentiment` column for comparison; it is a small demonstration file, not a benchmark evaluation set.
