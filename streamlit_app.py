import streamlit as st
import joblib
import os

MODEL_DIR = "models"
VECT = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL = os.path.join(MODEL_DIR, "best_model.joblib")

@st.cache_resource
def load_artifacts():
    v = joblib.load(VECT)
    m = joblib.load(MODEL)
    return v, m

def predict(t, v, m):
    X = v.transform([t])
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)[0]
        return m.predict(X)[0], p
    return m.predict(X)[0], None

def main():
    st.title("Fake News Detector")
    text = st.text_area("Enter news text")
    if st.button("Predict"):
        if not text.strip():
            st.error("No text entered")
        else:
            v, m = load_artifacts()
            pred, proba = predict(text, v, m)
            label = "REAL" if pred == 1 else "FAKE"
            st.subheader(label)
            if proba is not None:
                st.write("REAL:", proba[1])
                st.write("FAKE:", proba[0])

if __name__ == "__main__":
    main()
