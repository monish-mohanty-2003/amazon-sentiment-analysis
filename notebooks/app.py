import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# SETTINGS
# ----------------------------

BASE_MODEL = "distilbert-base-uncased"

CHECKPOINT_PATH = r"D:\Masters\Python\CaseStudy\amazon-sentiment-analysis\notebooks\results\checkpoint-19432"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD TOKENIZER + MODEL
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH
)

model.to(device)
model.eval()

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title(" DistilBERT Sentiment Analyzer")
st.write("Fine-tuned on Amazon Reviews")

text = st.text_area("Enter your review:")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        if prediction == 1:
            st.success(f"Positive  (Confidence: {confidence:.2f})")
        else:
            st.error(f"Negative  (Confidence: {confidence:.2f})")
