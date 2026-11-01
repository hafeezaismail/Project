
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "hafeezaismail/banking77-distilbert-intent-classifier"

st.set_page_config(
    page_title="Banking77 Intent Classifier",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

st.title("🏦 Banking Customer Query Classifier")
st.write(
    "Enter a banking-related customer query. "
    "The fine-tuned DistilBERT model will predict its intent."
)

try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

query = st.text_area(
    "Enter customer query:",
    placeholder="Example: My card payment was declined."
)

if st.button("Predict Intent"):
    if query.strip():
        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_id = probabilities.argmax(dim=-1).item()
            confidence = probabilities[0][predicted_id].item()

        intent = model.config.id2label[predicted_id]

        st.subheader("Prediction Result")
        st.write("**Predicted Intent:**", intent)
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.progress(confidence)
    else:
        st.warning("Please enter a customer query.")
