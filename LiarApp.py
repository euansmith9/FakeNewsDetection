import numpy as np
import tensorflow as tf
import streamlit as st
from transformers import BertTokenizerFast
from lime.lime_text import LimeTextExplainer
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Constants
MODEL_PATH = hf_hub_download(
    repo_id="euansmith9/liar_bert_model",  
    filename="liar_bert.keras"           
)
TOKENIZER_NAME = "bert-base-uncased"
MAX_LEN = 128
THRESHOLD = 0.5                
ABSTAIN_MARGIN = 0.05          
CLASS_NAMES = ["Fake", "Real"] 
TOP_K = 5
LIME_SAMPLES = 1000

st.set_page_config(page_title="Fake News Detection Web App", layout="centered")
st.title("Short Statement - Fake News Detection")
st.subheader("This app utilises a BERT model to classify short political statements as real or fake. " \
"Disclaimer: This is a demo app and may not be accurate.")

# Load the pre-trained model and tokenizer
@st.cache_resource(show_spinner=False)
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

@st.cache_resource(show_spinner=False)
def get_explainer():
    return LimeTextExplainer(class_names=CLASS_NAMES, random_state=0)

def predict_proba_batch(texts):
    tokenizer = get_tokenizer()
    model = get_model()

    enc = tokenizer(
        texts,
        truncation=True, padding=True, max_length=MAX_LEN,
        return_tensors="tf"
    )

    inputs = {
        "token_ids": tf.cast(enc["input_ids"], tf.int32),
        "segment_ids": tf.cast(enc["token_type_ids"], tf.int32),
        "padding_mask": tf.cast(enc["attention_mask"], tf.int32),
    }

    logits = model(inputs, training=False)

    logits = tf.convert_to_tensor(logits)

    logits.shape[-1] == 1
    p_real = tf.sigmoid(logits[..., 0])
    p_fake = 1.0 - p_real
    probs = tf.stack([p_fake, p_real], axis=-1)

    return probs.numpy()

# Function to predict label and class index
def predict_label_and_class(text: str):
    probs = predict_proba_batch([text])[0]      
    p_fake, p_real = float(probs[0]), float(probs[1])

    if abs(p_real - THRESHOLD) <= ABSTAIN_MARGIN:
        label = "Unable to determine"
    else:
        label = "Real" if p_real >= THRESHOLD else "Fake"

    class_idx = int(np.argmax(probs))             
    return label, class_idx, p_fake, p_real

# Function to explain top factors using LIME
def explain_top_factors(text: str, class_idx: int, top_k: int = TOP_K):
    explainer = get_explainer()
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda xs: predict_proba_batch(xs),
        num_features=top_k,
        labels=[class_idx],
        num_samples=LIME_SAMPLES,
    )
    tuples = exp.as_list(label=class_idx)
    return [(tok.replace(" ##","").replace("##",""), float(w)) for tok, w in tuples[:top_k]]

# UI setup
text = st.text_area(
    "Paste a claim:",
    height=120,
    placeholder="Insert text here..."
)

if st.button("Check", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Scoring..."):
        label, class_idx, p_fake, p_real = predict_label_and_class(text.strip())

    st.header(label)
    st.caption(f"Probability (Real): {p_real:.3f} • Probability (Fake): {p_fake:.3f}")

    with st.spinner("Explaining..."):
        factors = explain_top_factors(text.strip(), class_idx, top_k=TOP_K)

    st.subheader(f"Top {TOP_K} factors")
    st.table({
        "Token / phrase": [t for t, _ in factors],
        "Weight": [round(w, 4) for _, w in factors],
        "Effect": ["supports" if w > 0 else "opposes" for _, w in factors],
    })






