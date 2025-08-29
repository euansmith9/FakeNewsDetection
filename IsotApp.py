import re
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Constants
MODEL_PATH = "ISOT_LR_model.joblib" 
THRESHOLD = 0.5
TOP_K = 5
CLASS_NAMES = {0: "Fake", 1: "Real"}  # assumes 0/1 labels

# Stopwords for clean_text
stop_words = ENGLISH_STOP_WORDS

st.set_page_config(page_title="Fake News Detection Web App", layout="centered")
st.title("News Article - Fake News Detection")
st.subheader("This app utilises a logistic regression model to classify news articles as real or fake. " \
"Disclaimer: This is a demo app and may not be accurate.")

# Text preprocessing functions

# Remove Reuters prefix from the text
def remove_reuters_prefix(text: str) -> str:
    return re.sub(r'^[A-Z][A-Z\s\.,\-()]+\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)

# Clean text by removing punctuation, numbers, and stopwords
def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)     # remove punctuation
    text = re.sub(r'\d+', '', text)         # remove numbers
    text = text.lower()
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

# Apply all preprocessing steps
def preprocess(text: str) -> str:
    return clean_text(remove_reuters_prefix(text))

# Load the pre-trained model pipeline
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipe = load_pipeline()
tfidf = pipe.named_steps["tfidf"]
clf   = pipe.named_steps["clf"]  # LogisticRegression

# Function to predict label and contributions
def predict_and_contributions(text: str):

    txt = preprocess(text)

    # Probabilities (aligned to clf.classes_)
    probs = pipe.predict_proba([txt])[0]
    classes = clf.classes_                # e.g., array([0, 1])
    p_fake = float(probs[np.where(classes == 0)[0][0]])
    p_real = float(probs[np.where(classes == 1)[0][0]])

    # Decisiion logic
    pred_class = 1 if p_real >= THRESHOLD else 0
    label_str = CLASS_NAMES[pred_class]

    # Vectorize once with the same tfidf used in training
    X = tfidf.transform([txt])          
    nz = X.nonzero()[1]                  
    if nz.size == 0:
        return label_str, p_fake, p_real, []

    # Get contributions for non-zero features
    w = clf.coef_[0]                     
    contrib = X.multiply(w).toarray().ravel()[nz]
    feats   = tfidf.get_feature_names_out()[nz]

    # Top-K features by absolute contribution
    order = np.argsort(-np.abs(contrib))[:TOP_K]
    feats_top = feats[order]
    vals_top  = contrib[order]

    # Function to determine effect based on contribution and predicted class
    def effect(v, pred_cls):
        if v >= 0:
            return "Real"
        else:
            return "Fake"

    rows = [(feats_top[i], float(vals_top[i]), effect(vals_top[i], pred_class))
            for i in range(len(order))]

    return label_str, p_fake, p_real, rows

# UI setup

# Text input area
text = st.text_area(
    "Paste a news article:",
    height=120,
    placeholder="Insert text here..."
)

# Button to trigger prediction
if st.button("Check", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # Display spinner while processing
    with st.spinner("Scoring..."):
        label, p_fake, p_real, rows = predict_and_contributions(text.strip()) # Prediction and contributions

    st.header(label)
    st.caption(f"Probability (Real): {p_real:.5f}")
    st.caption(f"Probability (Fake): {p_fake:.5f}") # Display results

    # Display contributions if available
    with st.spinner("Computing top n-grams..."):
        if rows:
            st.subheader(f"Top {TOP_K} n-grams influencing the decision")
            st.table({
                "n-gram":       [r[0] for r in rows],
                "Contribution": [round(r[1], 4) for r in rows], 
                "Effect":       [r[2] for r in rows],         
            })
        else:
            st.subheader(f"Top {TOP_K} n-grams")
            st.write("No informative n-grams found in this input.")









