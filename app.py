# app.py 
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model
from PIL import Image
import os, json, pandas as pd
from fpdf import FPDF

# Config
MODEL_PATH = "cnn_scratch_train.keras"
CLASS_NAMES = ["COVID", "Pneumonia", "Normal"]
HISTORY_FILE = "predictions_history.json"
IMG_SIZE = (128, 128)

st.set_page_config("X-Ray Classifier", page_icon="🩻", layout="centered")

# Authentication(Hard Code)
USER_CREDENTIALS = {"admin": "12345", "doctor": "med2025"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# Load model
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# History 
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")

def create_pdf_report(entry):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Header with warning background
        pdf.set_fill_color(255, 240, 240)  # Light red background
        pdf.rect(10, 10, 190, 30, 'F')
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(200, 0, 0)  # Red text
        pdf.ln(5)
        pdf.cell(0, 10, "WARNING: AI-GENERATED MEDICAL ANALYSIS REPORT", ln=True, align="C")
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, "NOT FOR DIAGNOSTIC USE - REQUIRES PROFESSIONAL MEDICAL REVIEW", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_text_color(0, 0, 0)  # Black text
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 12, "Chest X-Ray AI Analysis Report", ln=True, align="C")
        pdf.ln(8)
        
        # Report generation info
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100)  # Gray text
        pdf.cell(0, 6, f"Generated on: {current_time}", ln=True, align="R")
        pdf.cell(0, 6, "System: AI X-Ray Classifier v1.0", ln=True, align="R")
        pdf.ln(5)
        
        # Patient Information Section
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 248, 255)  # Light blue background
        pdf.cell(0, 10, "PATIENT INFORMATION", ln=True, fill=True)
        pdf.ln(2)
        
        pdf.set_font("Arial", size=12)
        patient_info = [
            ("Patient Name:", entry.get('name', 'N/A')),
            ("Age:", f"{entry.get('age', 'N/A')} years" if entry.get('age') != 'N/A' else 'N/A'),
            ("Gender:", entry.get('gender', 'N/A')),
            ("Image Filename:", entry['filename']),
            ("Reported Symptoms:", entry.get('symptoms', 'N/A') if entry.get('symptoms') else 'N/A')
        ]
        
        for label, value in patient_info:
            pdf.cell(60, 8, label, ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, str(value), ln=True)
            pdf.set_font("Arial", size=12)
        
        pdf.ln(8)
        
        # AI Analysis Results Section
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(255, 248, 240)  # Light orange background
        pdf.cell(0, 10, "AI ANALYSIS RESULTS", ln=True, fill=True)
        pdf.ln(2)
        
        # Main prediction with confidence
        pdf.set_font("Arial", 'B', 16)
        prediction_color = {
            'COVID': (200, 0, 0),      # Red
            'Pneumonia': (200, 100, 0), # Orange  
            'Normal': (0, 150, 0)       # Green
        }
        color = prediction_color.get(entry['predicted'], (0, 0, 0))
        pdf.set_text_color(*color)
        pdf.cell(0, 12, f"Primary Classification: {entry['predicted']}", ln=True)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Confidence Level: {entry['confidence']}%", ln=True)
        pdf.ln(5)
        
        # Detailed probabilities
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Detailed Classification Probabilities:", ln=True)
        pdf.set_font("Arial", size=11)
        
        for cls, prob in entry["probs"].items():
            pdf.cell(40, 7, f"{cls}:", ln=False)
            pdf.cell(30, 7, f"{prob}%", ln=False)
            
            # Add visual bar representation using simple characters
            bar_length = int(prob / 5)  # Scale down for display
            bar_display = "|" * min(bar_length, 20)
            pdf.cell(0, 7, f"[{bar_display:20}]", ln=True)
        
        pdf.ln(10)
        
        # Critical Disclaimers Section
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(255, 235, 235)  # Light red background
        pdf.cell(0, 10, "IMPORTANT MEDICAL DISCLAIMERS", ln=True, fill=True)
        pdf.ln(2)
        
        disclaimers = [
            "* This report is generated by an Artificial Intelligence system and is NOT a medical diagnosis.",
            "* AI analysis should NEVER replace professional medical evaluation by qualified healthcare providers.",
            "* This tool is intended for educational and research purposes only.",
            "* False positives and false negatives are possible - clinical correlation is essential.",
            "* Emergency medical conditions require immediate professional medical attention.",
            "* The AI model has limitations and may not detect all pathological conditions.",
            "* This analysis should be used only as a supplementary tool alongside clinical judgment."
        ]
        
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(150, 0, 0)  # Dark red
        for disclaimer in disclaimers:
            pdf.cell(0, 6, disclaimer, ln=True)
        pdf.ln(5)
        
        # Professional Recommendation Section
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 255, 240)  # Light green background
        pdf.cell(0, 8, "PROFESSIONAL RECOMMENDATIONS", ln=True, fill=True)
        pdf.ln(2)
        
        recommendations = {
            'COVID': [
                "* Consult healthcare provider immediately for clinical evaluation",
                "* Follow current COVID-19 testing and isolation protocols",
                "* Monitor symptoms and seek emergency care if breathing difficulties occur"
            ],
            'Pneumonia': [
                "* Seek immediate medical attention for proper diagnosis and treatment",
                "* Professional chest X-ray interpretation and clinical correlation required",
                "* Blood tests and additional imaging may be necessary"
            ],
            'Normal': [
                "* While AI suggests normal findings, clinical evaluation is still recommended",
                "* Consult healthcare provider if symptoms persist or worsen",
                "* Regular medical check-ups remain important for overall health"
            ]
        }
        
        pdf.set_font("Arial", size=10)
        current_recommendations = recommendations.get(entry['predicted'], recommendations['Normal'])
        for rec in current_recommendations:
            pdf.cell(0, 6, rec, ln=True)
        
        pdf.ln(8)
        
        # Additional warning box
        pdf.set_fill_color(255, 220, 220)  # Light red background
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(150, 0, 0)
        pdf.cell(0, 8, "CRITICAL NOTICE:", ln=True, fill=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, "This AI system is experimental and for research purposes only.", ln=True)
        pdf.cell(0, 6, "Do NOT use for clinical diagnosis or treatment decisions.", ln=True)
        pdf.cell(0, 6, "Always consult qualified medical professionals for health concerns.", ln=True)
        pdf.ln(5)
        
        # Footer with contact and version info
        pdf.set_text_color(100, 100, 100)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, "This AI system is a research tool and should not be used for clinical decision-making.", ln=True, align="C")
        pdf.cell(0, 5, "For medical emergencies, contact your local emergency services immediately.", ln=True, align="C")
        pdf.cell(0, 5, f"Report ID: AI-{entry['filename']}-{current_time.replace(' ', '-').replace(':', '')}", ln=True, align="C")
        
        pdf_output = f"AI_XRay_Report_{entry['filename']}_{current_time.replace(' ', '_').replace(':', '')}.pdf"
        pdf.output(pdf_output)
        return pdf_output
        
    except Exception as e:
        st.error(f"Error creating enhanced PDF: {e}")
        return None

history = load_history()

# Streamlit UI
st.title("🩻 Chest X-Ray Classifier (COVID / Pneumonia / Normal)")

menu = st.sidebar.selectbox("Menu", ["Predict", "Batch Predict", "History", "Model Info"])

if menu == "Predict":
    st.header("Single Image Prediction")
    name = st.text_input("Patient Name")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms (comma-separated)")
    uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Prepare image for prediction
        img_resized = img.resize(IMG_SIZE)
        x = kimage.img_to_array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)
        
        # Make prediction
        with st.spinner("Analyzing X-ray..."):
            preds = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(np.max(preds)) * 100
        
        st.markdown(f"### Prediction: **{pred_label}** — Confidence: **{confidence:.2f}%**")

        probs_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability (%)": (preds * 100).round(2)})
        st.table(probs_df)

        # Save to history
        entry = {
            "filename": uploaded.name,
            "predicted": pred_label,
            "confidence": round(confidence, 2),
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "probs": {CLASS_NAMES[i]: float(round(float(preds[i] * 100), 2)) for i in range(len(CLASS_NAMES))}
        }
        history.insert(0, entry)
        save_history(history)
        st.success("Prediction saved to history.")

elif menu == "Batch Predict":
    st.header("Batch Prediction")
    uploaded_files = st.file_uploader("Upload multiple chest X-ray images", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        progress_bar = st.progress(0)
        
        for idx, uploaded in enumerate(uploaded_files):
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            img = Image.open(uploaded).convert("RGB")
            img_resized = img.resize(IMG_SIZE)
            x = kimage.img_to_array(img_resized) / 255.0
            x = np.expand_dims(x, axis=0)
            
            preds = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(np.max(preds)) * 100

            results.append({
                "Filename": uploaded.name,
                "Predicted Class": pred_label,
                "Confidence (%)": round(confidence, 2),
                "COVID (%)": round(float(preds[0] * 100), 2),
                "Pneumonia (%)": round(float(preds[1] * 100), 2),
                "Normal (%)": round(float(preds[2] * 100), 2)
            })

        progress_bar.empty()
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.success(f"Batch prediction complete for {len(uploaded_files)} images.")
        
        # Option to download results as CSV
        if results:
            csv = pd.DataFrame(results).to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="batch_prediction_results.csv",
                mime="text/csv"
            )

elif menu == "History":
    st.header("Prediction History")
    if not history:
        st.info("No predictions saved yet.")
    else:
        # Display history in a more organized way
        for idx, entry in enumerate(history):
            with st.expander(f"{entry['filename']} - {entry['predicted']} ({entry['confidence']}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Patient Information:**")
                    st.write(f"Name: {entry.get('name', 'N/A')}")
                    st.write(f"Age: {entry.get('age', 'N/A')}")
                    st.write(f"Gender: {entry.get('gender', 'N/A')}")
                    st.write(f"Symptoms: {entry.get('symptoms', 'N/A')}")
                
                with col2:
                    st.write("**Prediction Results:**")
                    st.write(f"Prediction: {entry['predicted']}")
                    st.write(f"Confidence: {entry['confidence']}%")
                    
                    probs_display = []
                    for cls, prob in entry['probs'].items():
                        probs_display.append(f"{cls}: {prob}%")
                    st.write("Probabilities: " + ", ".join(probs_display))
                
                # PDF download button (manual download only)
                if st.button(f"📄 Generate PDF Report", key=f"generate_{idx}"):
                    pdf_path = create_pdf_report(entry)
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label=f"📄 Download PDF Report",
                                data=pdf_file,
                                file_name=f"{entry['filename']}_report.pdf",
                                mime="application/pdf",
                                key=f"download_{idx}"
                            )
                        st.success(f"PDF report generated successfully!")
                    else:
                        st.error("Failed to generate PDF report.")
        
        # Clear history button
        st.divider()
        if st.button("🗑️ Clear All History", type="secondary"):
            history.clear()
            save_history(history)
            st.rerun()

elif menu == "Model Info":
    st.header("Model & Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Specifications")
        st.write(f"**Model Type:** Custom CNN")
        st.write(f"**Input Size:** {IMG_SIZE[0]} × {IMG_SIZE[1]} × 3")
        st.write(f"**Classes:** {len(CLASS_NAMES)}")
        st.write(f"**Class Names:** {', '.join(CLASS_NAMES)}")
    
    with col2:
        st.subheader("⚠️ Important Notes")
        st.write("• Model performance may vary with real-world data")
        st.write("• False negatives can be harmful")
        st.write("• Not for sole diagnostic use")
        st.write("• Educational/research purposes only")
    
    st.divider()
    
    st.subheader("🔬 Model Performance Guidelines")
    st.markdown("""
    **This model is designed for:**
    - Research and educational purposes
    - Assisting healthcare professionals in preliminary analysis
    - Demonstrating AI applications in medical imaging
    
    **This model should NOT be used for:**
    - Final medical diagnosis without professional review
    - Emergency medical decisions
    - Replacing professional medical consultation
    
    **Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.**
    """)
    
    # Display model summary if possible
    if st.checkbox("🔍 Show Model Architecture"):
        try:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            model.summary()
            sys.stdout = old_stdout
            model_summary = buffer.getvalue()
            st.code(model_summary, language="text")
        except Exception as e:
            st.error(f"Could not display model summary: {e}")

# Add logout button in sidebar
st.sidebar.divider()
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()