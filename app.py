import streamlit as st
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

# ---------------- CSS DESIGN ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 25px;
}

.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
}

.result {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">💬 Customer Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a review to predict sentiment (Positive / Negative)</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area("✍️ Enter your review:", height=120)

    predict_btn = st.button("🚀 Analyze Sentiment")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:

    if user_input.strip() != "":

        # Transform text
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)

        # Confidence score (optional)
        try:
            prob = model.predict_proba(transformed)
            confidence = max(prob[0]) * 100
        except:
            confidence = 0

        # ---------------- RESULT LOGIC ----------------
        # CHANGE THIS IF YOUR LABELS ARE OPPOSITE
        if prediction[0] == 1:
            label = "😊 Positive Sentiment"
            color = "#22c55e"
        else:
            label = "😞 Negative Sentiment"
            color = "#ef4444"

        # OUTPUT BOX
        st.markdown(f"""
        <div class="result" style="background-color:{color}33; color:{color};">
            {label}<br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a review")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Major Project | Customer Sentiment Analysis using NLP | Developed by Alok Pradhan")