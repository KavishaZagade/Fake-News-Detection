
import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

# Custom CSS for background and styling
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: 	#e5d0ff !important;  /* Light blue */
        height: 100%;
        margin: 0;
        padding: 0;
    }

   pips
    .stTextArea, .stButton {
        font-size: 18px;
        border-radius: 3px;
    }

    .title {
        font-size: 52px;
        font-weight: bold;
        color: #05714B;
        text-align: center;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        color: black;
        line-height: 1;
        padding: 17px;
        border: 2px solid black;
        background-color:white;
    }
    .subtext{
        font-size:18px;
        font-weight:600;
        font-family: sans-serif;
        margin-left:10px
        margin-bottom:25px;
        }
    </style>
""", unsafe_allow_html=True)

# Title in styled div
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Enter a News Article below to check whether it is Fake or Real.</div>', unsafe_allow_html=True)

# Input
news_input = st.text_area("", "")

# Button and Prediction
if st.button("🔍 Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction  = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is **Real**! ")
        else:
            st.error("❌ The News is **Fake**! ")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
        
st.markdown('</div>', unsafe_allow_html=True)
