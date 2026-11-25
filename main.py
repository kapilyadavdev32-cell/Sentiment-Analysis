from io import BytesIO

import pandas as pd
import requests
import streamlit as st

PREDICTION_ENDPOINT = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")
st.caption("For bulk predictions, upload a CSV or TSV with a 'Sentence' or 'verified_reviews' column.")

uploaded_file = st.file_uploader("Choose a CSV or TSV file", type=["csv", "tsv", "tab"])
user_input = st.text_input("Enter text", "")

if st.button("Predict"):
    try:
        if uploaded_file is not None:
            response = requests.post(
                PREDICTION_ENDPOINT,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            if "application/json" in response.headers.get("Content-Type", ""):
                st.error(response.json().get("error", "Prediction failed."))
            else:
                response.raise_for_status()
                csv_bytes = response.content
                result = pd.read_csv(BytesIO(csv_bytes))
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    label="Download predictions",
                    data=csv_bytes,
                    file_name="Predictions.csv",
                    mime="text/csv",
                )
        elif user_input.strip():
            response = requests.post(
                PREDICTION_ENDPOINT,
                data={"text": user_input.strip()},
                timeout=30,
            )
            result = response.json()
            if response.ok:
                st.write(f"Predicted sentiment: {result['prediction']}")
            else:
                st.error(result.get("error", "Prediction failed."))
        else:
            st.warning("Enter text or upload a CSV file first.")
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    except (ValueError, KeyError) as exc:
        st.error(f"Could not process the API response: {exc}")
