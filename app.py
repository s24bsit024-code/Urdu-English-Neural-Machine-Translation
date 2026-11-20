import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# Urdu ↔ English Neural Machine Translation App
# ============================================================

st.set_page_config(
    page_title="Urdu ↔ English Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Urdu ↔ English Translator")
st.write("A Deep Learning NLP translation application using fine-tuned Hugging Face models.")

# ------------------------------------------------------------
# IMPORTANT:
# Replace YOUR_HUGGINGFACE_USERNAME with your actual
# Hugging Face username after uploading the two models.
# ------------------------------------------------------------
HF_USERNAME = "YOUR_HUGGINGFACE_USERNAME"

UR_EN_REPO = f"{HF_USERNAME}/urdu-english-translation"
EN_UR_REPO = f"{HF_USERNAME}/english-urdu-translation"


@st.cache_resource
def load_model(repo_id):
    """Load tokenizer and fine-tuned translation model."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

    model.eval()
    return tokenizer, model


def translate(text, tokenizer, model):
    """Translate input text using the selected model."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------

direction = st.radio(
    "Select translation direction:",
    [
        "Urdu → English",
        "English → Urdu"
    ]
)

if direction == "Urdu → English":
    placeholder = "مثال: پاکستان ایک خوبصورت ملک ہے۔"
    source_label = "Enter Urdu text"
else:
    placeholder = "Example: Pakistan is a beautiful country."
    source_label = "Enter English text"

text = st.text_area(
    source_label,
    height=160,
    placeholder=placeholder
)

if st.button("🔄 Translate", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text first.")

    elif HF_USERNAME == "YOUR_HUGGINGFACE_USERNAME":
        st.error(
            "Please open app.py and replace "
            "'YOUR_HUGGINGFACE_USERNAME' with your actual "
            "Hugging Face username."
        )

    else:
        try:
            with st.spinner("Translating..."):

                if direction == "Urdu → English":
                    tokenizer, model = load_model(UR_EN_REPO)
                else:
                    tokenizer, model = load_model(EN_UR_REPO)

                result = translate(
                    text,
                    tokenizer,
                    model
                )

            st.success("Translation completed!")

            st.subheader("Translation")
            st.text_area(
                "Result",
                result,
                height=160
            )

        except Exception as e:
            st.error("Unable to load or use the model.")
            st.exception(e)


# ------------------------------------------------------------
# Project information
# ------------------------------------------------------------

with st.expander("About this project"):
    st.write(
        """
        This project is an NLP Neural Machine Translation system.

        Features:
        • Urdu → English translation
        • English → Urdu translation
        • Hugging Face pretrained models
        • Fine-tuned translation models
        • Streamlit web application
        """
    )

st.caption("Urdu ↔ English Neural Machine Translation | Deep Learning Project")
