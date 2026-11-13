# ------------------------------------------------------------
# AI-Based Customer Review & Sentiment Analysis
# Streamlit web app  |  Run from the project folder:
#     python -m streamlit run app/app.py
# ------------------------------------------------------------
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent.parent      # the sentiment_analysis folder
DATA_PATH = BASE / "data" / "reviews_clean.csv"
MODEL_PATH = BASE / "models" / "sentiment_model.pkl"

st.set_page_config(page_title="Customer Review Sentiment Analysis", page_icon="💬", layout="wide")

COLORS = {"Positive": "#2e7d32", "Neutral": "#757575", "Negative": "#c62828"}
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}

# ---------- Same text cleaning as notebook 02 ----------
KEEP_WORDS = {"not", "no", "nor", "never", "cannot", "very", "too", "least", "less", "more", "most"}
STOP_WORDS = ENGLISH_STOP_WORDS - KEEP_WORDS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 1)


# ---------- Load model and data once (cached) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Review Date"])


model = load_model()
df = load_data()


def predict(texts):
    """Return predicted sentiment and confidence (%) for a list of review texts."""
    cleaned = [clean_text(t) for t in texts]
    labels = model.predict(cleaned)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(cleaned).max(axis=1) * 100
    else:
        confidence = [None] * len(labels)
    return labels, confidence, cleaned


# ---------- Header ----------
st.title("💬 AI-Based Customer Review & Sentiment Analysis")
st.caption("India Superstore reviews · 2022–2025 · Python, Pandas, NLP (TF-IDF), Scikit-learn, Streamlit")

tab1, tab2, tab3 = st.tabs(["🔍 Analyse a Review", "📂 Analyse a File", "📈 Satisfaction Dashboard"])

# ============================================================
# TAB 1: ANALYSE ONE REVIEW
# ============================================================
with tab1:
    st.write("Type or paste a customer review. The model predicts whether it is **Positive**, **Neutral** or **Negative**.")

    examples = [
        "",
        "The chair is very comfortable and delivery was quick. Highly recommended!",
        "Worst phone ever, stopped working in two days. I want a refund.",
        "The table is okay, nothing special.",
        "Good product but delivery was late.",
        "Not happy with the packaging, the box was torn.",
    ]
    example = st.selectbox("Or pick an example", examples, format_func=lambda x: x if x else "— choose an example —")
    review = st.text_area("Customer review", value=example, height=120, placeholder="e.g. Very poor quality table, arrived damaged")

    if st.button("Analyse sentiment", type="primary"):
        if not review.strip():
            st.warning("Please type a review first.")
        else:
            label, conf, cleaned = predict([review])
            label, conf = label[0], conf[0]
            if not cleaned[0]:
                st.warning("The review has no meaningful words after cleaning. Try a longer review.")
            else:
                msg = f"{EMOJI[label]} **{label}**" + (f" · confidence {conf:.0f}%" if conf is not None else "")
                {"Positive": st.success, "Neutral": st.info, "Negative": st.error}[label](msg)

                if hasattr(model, "predict_proba"):
                    probs = pd.Series(model.predict_proba(cleaned)[0] * 100, index=model.classes_, name="Probability %")
                    st.bar_chart(probs)
                with st.expander("How the model sees this review"):
                    st.write("**Cleaned text:**", cleaned[0])
                    st.caption("Lowercased, punctuation and numbers removed, common filler words removed. "
                               "Words like 'not' and 'very' are kept because they change the meaning.")

# ============================================================
# TAB 2: ANALYSE A CSV FILE
# ============================================================
with tab2:
    st.write("Upload a CSV file with a column of review text. Every review gets a predicted sentiment, "
             "and you can download the results.")

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        text_cols = [c for c in data.columns if data[c].dtype == object or str(data[c].dtype) == "str"]
        if not text_cols:
            st.error("No text column found in this file.")
        else:
            default = text_cols.index("Review Text") if "Review Text" in text_cols else 0
            col = st.selectbox("Which column has the review text?", text_cols, index=default)
            data = data.dropna(subset=[col]).copy()

            labels, conf, _ = predict(data[col].astype(str).tolist())
            data["Predicted Sentiment"] = labels
            if conf[0] is not None:
                data["Confidence %"] = pd.Series(conf, index=data.index).round(1)

            counts = data["Predicted Sentiment"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reviews analysed", f"{len(data):,}")
            c2.metric("😊 Positive", f"{counts.get('Positive', 0) / len(data) * 100:.1f}%")
            c3.metric("😐 Neutral", f"{counts.get('Neutral', 0) / len(data) * 100:.1f}%")
            c4.metric("😞 Negative", f"{counts.get('Negative', 0) / len(data) * 100:.1f}%")

            st.bar_chart(counts)
            st.dataframe(data, hide_index=True)
            st.download_button("⬇️ Download results as CSV", data.to_csv(index=False).encode("utf-8"),
                               file_name="reviews_with_sentiment.csv", mime="text/csv")

    with st.expander("No file? Download a sample to try"):
        sample = df[["Review ID", "Review Text"]].sample(50, random_state=1)
        st.download_button("⬇️ Download sample_reviews.csv", sample.to_csv(index=False).encode("utf-8"),
                           file_name="sample_reviews.csv", mime="text/csv")

# ============================================================
# TAB 3: SATISFACTION DASHBOARD (from labelled reviews)
# ============================================================
with tab3:
    f1, f2 = st.columns(2)
    cats = f1.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    regions = f2.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    d = df[df["Category"].isin(cats) & df["Region"].isin(regions)].copy()

    if d.empty:
        st.warning("No reviews for these filters. Please select at least one category and one region.")
    else:
        pos = (d["Sentiment"] == "Positive").mean() * 100
        neg = (d["Sentiment"] == "Negative").mean() * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reviews", f"{len(d):,}")
        k2.metric("😊 Positive", f"{pos:.1f}%")
        k3.metric("😞 Negative", f"{neg:.1f}%")
        k4.metric("Satisfaction score", f"{pos - neg:.1f}", help="% positive minus % negative")

        st.subheader("Sentiment trend (3-month average, %)")
        monthly = pd.crosstab(d["Review Date"].dt.to_period("M"), d["Sentiment"], normalize="index") * 100
        monthly.index = monthly.index.to_timestamp()
        st.line_chart(monthly.rolling(3).mean().dropna(), color=[COLORS[c] for c in monthly.columns])

        a, b = st.columns(2)
        with a:
            st.subheader("Satisfaction score by year")
            d["Score"] = (d["Sentiment"] == "Positive").astype(int) - (d["Sentiment"] == "Negative").astype(int)
            yearly = d.groupby(d["Review Date"].dt.year)["Score"].mean() * 100
            yearly.index = yearly.index.astype(str)
            st.bar_chart(yearly)
        with b:
            st.subheader("% negative by sub-category")
            st.bar_chart(((d["Sentiment"] == "Negative").groupby(d["Sub-Category"]).mean() * 100)
                         .sort_values(ascending=False).head(8), horizontal=True)

        st.subheader("Top complaint reasons (% of negative reviews)")
        themes = {
            "Delivery delay": ["late", "delay", "week", "took", "waiting"],
            "Damaged / defective": ["damaged", "defective", "broken", "scratches", "stopped working"],
            "Poor quality": ["poor quality", "cheap", "flimsy", "worst"],
            "Packaging": ["packaging", "torn"],
            "Refund / return": ["refund", "return"],
            "Customer service": ["customer care", "respond", "response", "service"],
            "Price / value": ["price", "waste money", "overpriced", "expensive"],
        }
        negative = d[d["Sentiment"] == "Negative"]
        if len(negative):
            share = pd.Series({t: negative["Clean Text"].apply(lambda x: any(k in x for k in kws)).mean() * 100
                               for t, kws in themes.items()}).sort_values(ascending=False)
            st.bar_chart(share, horizontal=True, color=COLORS["Negative"])

        st.subheader("Word clouds")
        w1, w2 = st.columns(2)
        for col, sentiment, cmap in [(w1, "Positive", "Greens"), (w2, "Negative", "Reds")]:
            text = " ".join(d.loc[d["Sentiment"] == sentiment, "Clean Text"])
            with col:
                st.markdown(f"**{sentiment} reviews**")
                if text.strip():
                    wc = WordCloud(width=700, height=350, background_color="white",
                                   colormap=cmap, max_words=60).generate(text)
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.caption("No reviews for this selection.")