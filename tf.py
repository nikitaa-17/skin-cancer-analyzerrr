import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Skin Cancer Analyzer", layout="wide", page_icon="🩺")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .navbar {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid #374151;
    }
    div[role="radiogroup"] { display: flex; justify-content: space-around; }
    div[role="radiogroup"] label > div:first-child { display: none; }
    div[role="radiogroup"] label {
        padding: 10px 40px;
        border-radius: 12px;
        font-size: 18px;
        cursor: pointer;
        transition: 0.3s;
        color: #9ca3af;
    }
    div[role="radiogroup"] label[data-selected="true"] {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🩺 Skin Cancer Risk Analyzer")
st.info("Welcome to Cancer Risk Analyzer Tool.")

# --- NAVIGATION ---
st.markdown('<div class="navbar">', unsafe_allow_html=True)
tab = st.radio("", ["🏠 Home", "🧪 Tool", "👥 Team"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- IMAGE LOGIC ---
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

def analyze_image(image):
    img = np.array(image)
    img = cv2.resize(img, (256, 256))
    img = remove_hair(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    asymmetry = abs(w - h)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-5)
    color_std = np.std(img)
    diameter = max(w, h)
    annotated = img.copy()
    cv2.drawContours(annotated, [largest_contour], -1, (255,0,0), 2)
    return {
        "asymmetry": asymmetry, "circularity": circularity,
        "color_std": color_std, "diameter": diameter,
        "annotated": annotated, "thresh": thresh, "gray": gray
    }

def calculate_risk(f):
    score = (0.25 * (min(f["asymmetry"]/50, 1.0)) + 
             0.25 * (1 - f["circularity"]) + 
             0.25 * (min(f["color_std"]/50, 1.0)) + 
             0.25 * (min(f["diameter"]/150, 1.0)))
    return min(score * 100, 100)

# --- HOME ---
if tab == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Welcome to the Analyzer")
        st.write("This project uses **OpenCV** to analyze skin lesions based on morphological features.")
        if st.button("Start Analysis Now ➔"):
            st.info("Please navigate to the '🧪 Tool' tab.")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/Melanoma.jpg", caption="Malignant Melanoma Reference")

# --- TOOL ---
elif tab == "🧪 Tool":
    st.header("🧪 Skin Analysis Tool")
    uploaded_file = st.file_uploader("Upload a clear skin lesion image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        res = analyze_image(image)

        if res:
            risk_p = calculate_risk(res)
            st.markdown("### Diagnostic View")
            row1_col1, row1_col2 = st.columns([1, 1], gap="medium")
            
            with row1_col1:
                st.image(res["annotated"], caption="Detected Lesion Boundary", use_container_width=True)
            
            with row1_col2:
                st.metric("Estimated Risk Probability", f"{risk_p:.1f}%")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = risk_p,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#3b82f6"},
                             'steps': [{'range': [0, 35], 'color': "#2ecc71"},
                                       {'range': [35, 70], 'color': "#f1c40f"},
                                       {'range': [70, 100], 'color': "#e74c3c"}]}))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")
            row2_col1, row2_col2 = st.columns([1, 1], gap="large")

            with row2_col1:
                st.subheader("Structural Profile")
                categories = ['Asymmetry', 'Border Irreg.', 'Color Var.', 'Diameter']
                values = [res["asymmetry"]/50, 1-res["circularity"], res["color_std"]/50, res["diameter"]/150]
                fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#3b82f6'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        height=350, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig_radar, use_container_width=True)

            with row2_col2:
                st.subheader("Morphometric Observations")
                obs_asym = "High asymmetry detected" if res["asymmetry"] > 15 else "Symmetrical shape"
                obs_bord = "Irregular/Jagged borders" if res["circularity"] < 0.65 else "Smooth/Regular borders"
                obs_color = "High color variance" if res["color_std"] > 35 else "Uniform color distribution"
                risk_lvl = "LOW" if risk_p < 40 else "MODERATE" if risk_p < 70 else "HIGH"
                
                st.write(f"🧬 **Symmetry:** {obs_asym}")
                st.write(f"🧬 **Border:** {obs_bord}")
                st.write(f"🧬 **Color:** {obs_color}")
                
                if risk_lvl == "HIGH":
                    st.error(f"**Final Assessment:** HIGH Risk Profile Identified.")
                    st.info("💡 Clinical dermatological consultation is strongly advised.")
                elif risk_lvl == "MODERATE":
                    st.warning(f"**Final Assessment:** MODERATE Risk Profile Identified.")
                else:
                    st.success(f"**Final Assessment:** LOW Risk Profile Identified.")

                report_text = f"SKIN CANCER ANALYSIS REPORT\nDate: {datetime.now()}\nResult: {risk_lvl} ({risk_p:.1f}%)"
                st.download_button("📥 Download Report", data=report_text, file_name=f"Report_{datetime.now().strftime('%Y%m%d')}.txt")

            with st.expander("View Image Processing Pipeline"):
                ts1, ts2 = st.columns(2)
                ts1.image(res["gray"], caption="Grayscale Output")
                ts2.image(res["thresh"], caption="Otsu Segmentation")

# --- TEAM ---
elif tab == "👥 Team":
    st.header("Project Contributors")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.subheader("Nikita Khandare")
    with t2:
        st.subheader("Dr. Kushagra kashyap")
