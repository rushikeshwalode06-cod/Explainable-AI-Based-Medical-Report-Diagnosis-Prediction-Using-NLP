import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import nltk

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediScan AI — Explainable Diagnosis Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# NLTK setup (stopwords + lemmatizer, cached so it only downloads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub("[^a-z ]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------------------------
# Load model + vectorizer (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("medical_model.pkl")
    tfidf = joblib.load("metfidf.pkl")
    return model, tfidf

model, tfidf = load_artifacts()
class_names = list(model.classes_)

def predict_proba(texts):
    clean = [clean_text(t) for t in texts]
    vec = tfidf.transform(clean)
    probs = model.predict_proba(vec)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return probs / row_sums

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f4f9fb 0%, #eef3f8 100%);
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 50%, #2563eb 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(8, 145, 178, 0.25);
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin: 0;
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.05);
        margin-bottom: 1.2rem;
    }

    .result-badge {
        display: inline-block;
        padding: 0.55rem 1.3rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #0891b2, #2563eb);
        color: white;
        font-weight: 600;
        font-size: 1.15rem;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    }

    .confidence-tag {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 0.5rem;
    }

    .highlight-word {
        padding: 2px 5px;
        border-radius: 6px;
        margin: 0 1px;
        font-weight: 600;
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #0f172a;
        font-size: 1.15rem;
        margin-bottom: 0.6rem;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2rem;
    }

    .stButton>button {
        background: linear-gradient(135deg, #0891b2, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 MediScan AI</h1>
        <p>Explainable AI-based Medical Report Diagnosis Prediction using NLP —
        paste a radiology / clinical report below and get an instant, transparent prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ About this tool")
    st.write(
        "This app uses a **Random Forest** classifier trained on TF-IDF "
        "features extracted from clinical report text to predict a likely "
        "diagnosis, and explains *why* using **LIME**."
    )
    st.markdown("---")
    st.markdown("### 🧪 Try a sample report")
    samples = {
        "Kidney Stones": "Kidneys show multiple calculi causing mild hydronephrosis on the right side.",
        "Tuberculosis": "Chest CT demonstrates cavitary lesions in the right upper lobe with surrounding fibrosis, suggestive of active infection.",
        "Liver Cancer": "CT abdomen shows multiple liver lesions suspicious for malignancy with irregular margins.",
        "Normal": "The lungs are clear. Heart size is normal. No acute cardiopulmonary abnormality.",
        "Pneumonia": "Large opacity in the right upper lobe. Patient has fever and productive cough.",
    }
    chosen_sample = st.selectbox("Pick a sample report", ["-- none --"] + list(samples.keys()))
    st.markdown("---")
    st.markdown("### 📋 Supported conditions")
    st.write(", ".join(sorted(class_names)))
    st.markdown("---")
    st.caption("⚠️ For educational / research demonstration purposes only. Not a substitute for professional medical diagnosis.")

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Enter Medical Report</div>', unsafe_allow_html=True)

    default_text = samples[chosen_sample] if chosen_sample != "-- none --" else ""
    report_text = st.text_area(
        "Paste the clinical / radiology report text",
        value=default_text,
        height=200,
        placeholder="e.g. Chest X-ray shows patchy consolidation in the left lower lobe with air bronchograms...",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        predict_clicked = st.button("🔍 Predict Diagnosis", use_container_width=True)
    with col_b:
        explain_toggle = st.toggle("Show explainability (LIME)", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction + explanation
# ---------------------------------------------------------------------------
if predict_clicked:
    if not report_text.strip():
        st.warning("Please enter or select a report before predicting.")
    else:
        cleaned = clean_text(report_text)
        vec = tfidf.transform([cleaned])
        probs = model.predict_proba(vec)[0]
        probs = probs / probs.sum()  # safety normalization in case of library version drift
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = probs[pred_idx] * 100

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="result-badge">{pred_label}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="confidence-tag">Model confidence: <b>{confidence:.1f}%</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Probability bar chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Class Probabilities</div>', unsafe_allow_html=True)
            order = np.argsort(probs)[::-1]
            fig = go.Figure(
                go.Bar(
                    x=[probs[i] * 100 for i in order],
                    y=[class_names[i] for i in order],
                    orientation="h",
                    marker=dict(
                        color=[probs[i] * 100 for i in order],
                        colorscale=[[0, "#93c5fd"], [1, "#0891b2"]],
                    ),
                    text=[f"{probs[i]*100:.1f}%" for i in order],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Probability (%)",
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=10, r=30, t=10, b=30),
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=13),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if explain_toggle:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🧠 Why did the model predict this? (LIME explanation)</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Generating explanation..."):
                try:
                    from lime.lime_text import LimeTextExplainer

                    explainer = LimeTextExplainer(class_names=class_names)
                    exp = explainer.explain_instance(
                        cleaned,
                        predict_proba,
                        labels=[pred_idx],
                        num_features=10,
                    )
                    weights = dict(exp.as_list(label=pred_idx))

                    # Highlight words directly in the (cleaned) report text
                    max_w = max(abs(v) for v in weights.values()) if weights else 1
                    tokens = cleaned.split()
                    html_tokens = []
                    for tok in tokens:
                        if tok in weights:
                            w = weights[tok]
                            intensity = min(abs(w) / max_w, 1.0)
                            if w > 0:
                                color = f"rgba(16, 185, 129, {0.15 + 0.55*intensity})"
                            else:
                                color = f"rgba(239, 68, 68, {0.15 + 0.55*intensity})"
                            html_tokens.append(
                                f'<span class="highlight-word" style="background:{color}">{tok}</span>'
                            )
                        else:
                            html_tokens.append(tok)
                    st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

                    st.write("")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**🟢 Supports this diagnosis**")
                        pos = sorted([(k, v) for k, v in weights.items() if v > 0], key=lambda x: -x[1])
                        for word, w in pos:
                            st.write(f"`{word}` → +{w:.3f}")
                    with c2:
                        st.markdown("**🔴 Against this diagnosis**")
                        neg = sorted([(k, v) for k, v in weights.items() if v < 0], key=lambda x: x[1])
                        for word, w in neg:
                            st.write(f"`{word}` → {w:.3f}")

                except Exception as e:
                    st.info(
                        "LIME explanation could not be generated "
                        f"({e}). Showing top TF-IDF terms instead."
                    )
                    feature_names = tfidf.get_feature_names_out()
                    row = vec.toarray()[0]
                    top_idx = np.argsort(row)[::-1][:10]
                    for i in top_idx:
                        if row[i] > 0:
                            st.write(f"`{feature_names[i]}` — TF-IDF weight: {row[i]:.3f}")

            st.markdown("</div>", unsafe_allow_html=True)
else:
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        st.write("Enter a report on the left and click **Predict Diagnosis** to see results here.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit · Random Forest + TF-IDF · Explainability via LIME · '
    "For research/educational demonstration only, not for clinical use.</div>",
    unsafe_allow_html=True,
)
