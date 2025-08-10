import os
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from huggingface_hub import login
from transformers import pipeline

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="HealthGuard AI", layout="wide")
load_dotenv()

API_KEY = os.getenv("API_KEY")  # Google Gemini
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Hugging Face

if not API_KEY and not HF_TOKEN:
    st.error("⚠ Missing API_KEY and HUGGINGFACE_TOKEN in .env file.")
    st.stop()

# Setup Google Gemini AI
if API_KEY:
    genai.configure(api_key=API_KEY)

# Login to Hugging Face if token available
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        hf_chat = pipeline("text-generation", model="sshleifer/tiny-gpt2", device_map="auto")
    except Exception as e:
        st.warning(f"⚠ Could not load HF fallback model: {e}")
        hf_chat = None
else:
    hf_chat = None

# =========================================
# THEME SWITCHER
# =========================================
theme = st.sidebar.radio("🎨 Theme", ["Green & White", "Red & White"])

if theme == "Green & White":
    primary_color = "#004d00"
    secondary_color = "#e6ffe6"
    bubble_color = "#ccffcc"
    label_bg = "#d1f7d6"
else:
    primary_color = "#800000"
    secondary_color = "#ffe6e6"
    bubble_color = "#ffcccc"
    label_bg = "#ffd6d6"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: #ffffff;
    }}
    h1, h2, h3, .stMarkdown {{
        color: {primary_color};
    }}
    .section-title {{
        background-color: {secondary_color};
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: bold;
    }}
    div[data-testid="stChatMessage"] div[role="presentation"]:has(div[style*="user"]) {{
        background-color: {secondary_color};
        border-radius: 10px;
        padding: 10px;
    }}
    div[data-testid="stChatMessage"] div[role="presentation"]:has(div[style*="assistant"]) {{
        background-color: {bubble_color};
        border-radius: 10px;
        padding: 10px;
    }}
    button {{
        background-color: {primary_color} !important;
        color: white !important;
        border-radius: 10px;
    }}
    button:hover {{
        background-color: #333 !important;
    }}
    .label-box {{
        background-color: {label_bg};
        padding: 5px 10px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        color: {primary_color};
    }}
    </style>
""", unsafe_allow_html=True)

# =========================================
# ML MODEL
# =========================================
@st.cache_resource
def train_health_model():
    data = pd.read_csv("diabetes.csv")
    data = data[["Age", "BMI", "Glucose", "BloodPressure", "Outcome"]]
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return model, report

model, ml_report = train_health_model()

# =========================================
# FUNCTIONS
# =========================================
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text

def gemini_chat(prompt):
    if API_KEY:
        try:
            model_g = genai.GenerativeModel("gemini-1.5-pro")
            response = model_g.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini error: {e}")
    if hf_chat:
        try:
            hf_response = hf_chat(prompt, max_length=200, num_return_sequences=1)
            return hf_response[0]['generated_text']
        except Exception as hf_err:
            return f"⚠ Both Gemini and HF fallback failed: {hf_err}"
    return "⚠ No AI service available."

def predict_disease(user_data):
    df = pd.DataFrame([user_data])
    risk_score = model.predict_proba(df)[0][1] * 100
    prediction = model.predict(df)[0]
    status = "High Risk" if prediction == 1 else "Low Risk"
    mitigation = []
    if user_data["BMI"] > 30:
        mitigation.append("Consider a daily exercise routine and a balanced diet.")
    if user_data["Glucose"] > 126:
        mitigation.append("Limit sugar intake, monitor carbs, and get blood sugar tests regularly.")
    if user_data["BloodPressure"] > 140:
        mitigation.append("Reduce salt intake, manage stress, and check BP weekly.")
    return {
        "status": status,
        "risk_score": round(risk_score, 2),
        "mitigation": mitigation if mitigation else ["Maintain current healthy habits."]
    }

# =========================================
# UI
# =========================================
st.markdown(f"<h1 style='color:{primary_color};'>🩺 HealthGuard AI – Your Personal Health Assistant</h1>", unsafe_allow_html=True)
st.write("Chat with an AI assistant, upload your medical report, and get personalized health insights.")

# Sidebar - PDF upload
st.sidebar.header("📄 Upload Medical Report")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
pdf_text = ""
if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.sidebar.success("PDF uploaded & processed ✅")

# Chat Section
st.markdown("<div class='section-title'>💬 Chat with HealthGuard AI</div>", unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your health question here...")
if user_input:
    full_prompt = f"Medical Report:\n{pdf_text}\n\nUser Question: {user_input}" if pdf_text else f"User Question: {user_input}"
    bot_reply = gemini_chat(full_prompt)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", bot_reply))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Prediction Section
st.markdown("<div class='section-title'>📊 Predict Your Health Risk</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="label-box">Age</div>', unsafe_allow_html=True)
    age = st.number_input("", 1, 120, 30)
with col2:
    st.markdown('<div class="label-box">BMI</div>', unsafe_allow_html=True)
    bmi = st.number_input("", 10.0, 50.0, 22.0)
with col3:
    st.markdown('<div class="label-box">Glucose Level</div>', unsafe_allow_html=True)
    glucose = st.number_input("", 50, 300, 90)
with col4:
    st.markdown('<div class="label-box">Blood Pressure</div>', unsafe_allow_html=True)
    bp = st.number_input("", 80, 200, 120)

if st.button("Predict Health Risk"):
    result = predict_disease({
        "Age": age, "BMI": bmi, "Glucose": glucose, "BloodPressure": bp
    })
    st.markdown(f"*Risk Status:* {result['status']}")
    st.markdown(f"*Risk Score:* {result['risk_score']}%")
    st.markdown("*Mitigation Steps:*")
    for step in result["mitigation"]:
        st.write(f"✅ {step}")
    st.session_state.latest_report = {
        "chat_history": st.session_state.chat_history,
        "prediction": result
    }

with st.expander("📈 View Model Accuracy Report"):
    st.json(ml_report)

# =========================================
# DOWNLOAD REPORT FEATURE
# =========================================
if "latest_report" in st.session_state:
    report_content = "🩺 HealthGuard AI – Health Report\n\n"
    report_content += "=== Chat History ===\n"
    for role, message in st.session_state.latest_report["chat_history"]:
        report_content += f"{role.capitalize()}: {message}\n"
    report_content += "\n=== Prediction Result ===\n"
    report_content += f"Risk Status: {st.session_state.latest_report['prediction']['status']}\n"
    report_content += f"Risk Score: {st.session_state.latest_report['prediction']['risk_score']}%\n"
    report_content += "Mitigation Steps:\n"
    for step in st.session_state.latest_report["prediction"]["mitigation"]:
        report_content += f"- {step}\n"

    st.download_button(
        label="📥 Download My Report",
        data=report_content,
        file_name="healthguard_report.txt",
        mime="text/plain"
    )
