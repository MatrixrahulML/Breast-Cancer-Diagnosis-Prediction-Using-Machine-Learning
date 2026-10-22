import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Breast Cancer ML Research",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Prediction - ML Research By MatrixRahul")
st.warning(
    "Educational demonstration only. "
    "This model is not validated for clinical diagnosis."
)

model = joblib.load("breast_cancer_model.joblib")

with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

features = metadata["features"]
threshold = float(metadata["default_threshold"])

st.write(
    "Upload a CSV containing the 30 measurement columns "
    "used by the trained model."
)

uploaded_file = st.file_uploader(
    "Upload measurement CSV",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    missing = [c for c in features if c not in data.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X_input = data[features].apply(
            pd.to_numeric, errors="coerce"
        )

        # Reject non-empty values that could not be converted.
        invalid = (
            data[features].notna()
            & X_input.isna()
        ).any().any()

        if invalid:
            st.error("Some measurement values are not numeric.")
        else:
            probability = model.predict_proba(X_input)[:, 1]
            prediction = (probability >= threshold).astype(int)

            results = pd.DataFrame({
                "Estimated_Malignant_Probability": probability,
                "Model_Class": np.where(
                    prediction == 1, "Malignant", "Benign"
                )
            })

            st.subheader("Model output")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download predictions",
                results.to_csv(index=False),
                file_name="prediction_results.csv",
                mime="text/csv"
            )