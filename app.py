import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from predict import predict_image, DEVICE

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ATPFv2 Oral Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PROFESSIONAL CSS
# ==========================================================

st.markdown("""
<style>
/* ================= RESULTS ================= */
.result-card{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0 12px 30px rgba(0,0,0,.08);
    margin-bottom:20px;
    transition:0.35s;
}

.result-card:hover{
    transform:translateY(-6px);
    box-shadow:0 20px 45px rgba(37,99,235,.20);
}

.result-title{
    font-size:22px;
    font-weight:700;
    color:#0F172A;
    margin-bottom:15px;
}

/* ===================== SIDEBAR CARD ===================== */
.side-card{
    background:linear-gradient(135deg, #2563EB, #4F46E5);
    padding:18px;
    border-radius:18px;
    margin-bottom:18px;
    box-shadow:0px 10px 30px rgba(0,0,0,.18);
    transition:0.35s;
    color:white;
}

.side-card:hover{
    transform:translateY(-5px);
    box-shadow:0px 20px 40px rgba(37,99,235,.35);
}

.side-card h4{
    margin:0;
    color:#E0E7FF;
    font-size:14px;
}

.side-card h2{
    margin-top:8px;
    margin-bottom:4px;
    font-size:26px;
}

.side-card p{
    margin:0;
    font-size:13px;
    color:#D1D5DB;
}            

/* ---------------- Main App ---------------- */
.stApp{
    background:linear-gradient(180deg, #EEF4FF, #F8FAFC, #FFFFFF);
}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #0F172A, #1E293B, #111827);
    padding:18px;
    border-radius:18px;
    box-shadow:0 10px 30px rgba(0,0,0,.08);
    transition:.35s;
}

[data-testid="metric-container"]:hover{
    transform:translateY(-6px);
    box-shadow:0 20px 40px rgba(37,99,235,.25);
}

/* ---------------- Main Container ---------------- */
.block-container{
    max-width:1450px;
    padding-top:4rem;
    padding-bottom:2rem;
}
            
/* ================= HERO ================= */
.hero{
    background:linear-gradient(135deg, #2563EB, #4F46E5, #7C3AED);
    padding:45px;
    border-radius:30px;
    text-align:center;
    margin-bottom:35px;
    box-shadow:0px 20px 50px rgba(0,0,0,.18);
    animation:fadeIn 1s;
}

.hero-icon{
    font-size:70px;
    margin-bottom:10px;
}

.hero-title{
    font-size:42px;
    font-weight:800;
    color:white;
}

.hero-subtitle{
    font-size:19px;
    margin-top:15px;
    color:#E2E8F0;
    line-height:1.6;
}

@keyframes fadeIn{
    0%{ opacity:0; transform:translateY(-30px); }
    100%{ opacity:1; transform:translateY(0); }
}            

/* ---------------- Title ---------------- */
.title{
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:#0F172A;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#64748B;
    margin-bottom:20px;
}

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"]{
    background:#0F172A;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label{
    color:white;
}

/* ---------------- File Upload ---------------- */
[data-testid="stFileUploader"]{
    background:white;
    border-radius:15px;
    padding:15px;
    border:2px dashed #3B82F6;
}

/* ---------------- Images ---------------- */
img{
    border-radius:20px;
    transition:.4s;
    box-shadow:0 10px 35px rgba(0,0,0,.12);
}

img:hover{
    transform:scale(1.02);
}

/* ---------------- Buttons ---------------- */
.stButton>button{
    background:#2563EB;
    color:white;
    border-radius:12px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

/* ---------------- Metric ---------------- */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

/* ---------------- Horizontal Line ---------------- */
hr{
    margin-top:25px;
    margin-bottom:25px;
}

/* ================= PREMIUM UPLOAD BOX ================= */
.upload-box{
    background:white;
    padding:35px;
    border-radius:25px;
    box-shadow:0px 15px 40px rgba(0,0,0,.08);
    text-align:center;
    margin-bottom:30px;
    transition:0.4s;
}

.upload-box:hover{
    transform:translateY(-5px);
    box-shadow:0px 25px 60px rgba(37,99,235,.15);
}

.upload-icon{
    font-size:70px;
    margin-bottom:10px;
}

.upload-title{
    font-size:28px;
    font-weight:700;
    color:#0F172A;
    margin-bottom:8px;
}

.upload-subtitle{
    font-size:16px;
    color:#64748B;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <div class="hero-icon">🧬</div>
    <div class="hero-title">ATPFv2 Oral Cancer Histopathological AI Platform</div>
    <div class="hero-subtitle">
        Artificial Intelligence Assisted Tissue Segmentation <br>
        using Deep Learning and Adaptive Tissue Pattern Fusion (ATPFv2)
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("""
<div style="text-align:center;padding:15px;">
    <h1 style="color:white; font-size:28px; margin-bottom:5px;">🧬 ATPF AI</h1>
    <p style="color:#CBD5E1; font-size:15px;">Research Edition v2.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Model Card
st.sidebar.markdown("""
<div class="side-card">
    <h4>🧠 Model</h4>
    <h2>ATPFv2</h2>
    <p>Adaptive Tissue Pattern Fusion</p>
</div>
""", unsafe_allow_html=True)

# Encoder Card
st.sidebar.markdown("""
<div class="side-card">
    <h4>⚙ Encoder</h4>
    <h2>ResNet34</h2>
    <p>ImageNet Pretrained</p>
</div>
""", unsafe_allow_html=True)

# Device Card
st.sidebar.markdown(f"""
<div class="side-card">
    <h4>🚀 Device</h4>
    <h2>{DEVICE.type.upper()}</h2>
    <p>Inference Engine</p>
</div>
""", unsafe_allow_html=True)

# Input Card
st.sidebar.markdown("""
<div class="side-card">
    <h4>🖼 Input Size</h4>
    <h2>512 × 512</h2>
    <p>RGB Histopathology</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📋 Workflow
✅ Upload Image  
⬇  
🧹 Preprocess  
⬇  
🧠 ATPFv2 Analysis  
⬇  
🎯 Segmentation  
⬇  
📊 Visualization  
⬇  
📄 Download Report  
""")

st.sidebar.markdown("---")

st.sidebar.info("""
⚠ Research Use Only

This software assists in oral cancer
histopathological image segmentation.

It should not replace expert pathological diagnosis.
""")

# ==========================================
# IMAGE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg", "tif"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.success("✅ Image uploaded successfully.")
    image = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🔬 Analyze Histopathological Image"):
        # Progress Bar
        progress = st.progress(0)
        status = st.empty()
        steps = [
            "📥 Reading image...",
            "🧹 Preprocessing...",
            "🧠 Extracting tissue features...",
            "🔬 Running ATPFv2 model...",
            "🎯 Generating segmentation..."
        ]
        
        for i, step in enumerate(steps):    
            status.info(step)
            progress.progress((i + 1) * 20)
            time.sleep(0.25)
            
        # Prediction
        start = time.time()
        probability, mask, confidence, area = predict_image(image)
        
        # Clear progress indicators
        progress.empty()
        status.empty()
        
        inference_time = time.time() - start
        
        # Original Image processing
        original = np.array(image)
        original = cv2.resize(original, (512, 512))
        
        # Heatmap
        heatmap = cv2.applyColorMap(probability, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = original.copy()
        red = np.zeros_like(original)
        red[:, :, 2] = 255
        overlay[mask > 0] = (0.6 * original[mask > 0] + 0.4 * red[mask > 0]).astype(np.uint8)
        
        st.markdown("""
        <div class="result-card">
            <div class="result-title">📊 AI Analysis Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            st.metric("🎯 Confidence", f"{confidence*100:.2f}%")
        with c2:
            st.metric("🧬 Tumor Area", f"{area:.2f}%")
        with c3:
            st.metric("⚡ Inference", f"{inference_time:.3f} sec")
        with c4:
            device_name = "CUDA" if predict_image.__globals__["DEVICE"].type == "cuda" else "CPU"
            st.metric("💻 Device", device_name)
            
        st.write("---") 
        st.markdown("## 📊 AI Analysis Results") 
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🖼 Original Image")
            st.image(original, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)                                                                                    
            
        with row1_col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🎨 Segmentation Overlay")
            st.image(overlay, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.write("")
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🌡 Probability Heatmap")
            st.image(heatmap, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with row2_col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🧩 Binary Mask")
            st.image(mask, use_container_width=True)                                                                  
            st.markdown("</div>", unsafe_allow_html=True)
            
        # ==========================================
        # Prediction Summary
        # ==========================================
        st.write("")
        st.markdown(f"""
        <div class="result-card">
            <h3>📋 Prediction Summary</h3>                                            
            <b>Model:</b> ATPFv2<br><br>
            <b>Prediction Confidence:</b> {confidence*100:.2f}%<br><br>
            <b>Segmented Area:</b> {area:.2f}%<br><br>
            <b>Inference Time:</b> {inference_time:.3f} sec
        </div>
        """, unsafe_allow_html=True)
        
        # Fixed logic flow for confidence alerts
        if confidence > 0.90:
            st.success("🟢 High Confidence Prediction")   
        elif confidence > 0.70:
            st.warning("🟡 Moderate Confidence Prediction")
        else:
            st.error("🔴 Low Confidence Prediction")
            
        st.write("")
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            _, mask_buffer = cv2.imencode(".png", mask)
            st.download_button(
                label="📥 Download Binary Mask",
                data=mask_buffer.tobytes(),
                file_name="Predicted_Mask.png",
                mime="image/png"
            )
            
        with col_download2:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)     
            _, overlay_buffer = cv2.imencode(".png", overlay_bgr)
            st.download_button(
                label="📥 Download Overlay",
                data=overlay_buffer.tobytes(),
                file_name="Overlay_Result.png",
                mime="image/png"    
            )
            
        report = f"""=========================================
Model               : ATPFv2
Encoder             : ResNet34
Confidence          : {confidence*100:.2f} %
Segmented Area      : {area:.2f} %
Inference Time      : {inference_time:.3f} sec
Input Size          : 512 x 512
=========================================
This report is generated automatically by
the ATPFv2 Oral Cancer AI Platform.
For research and educational purposes only.
========================================="""
        
        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="Prediction_Report.txt",
            mime="text/plain"
        )
        
        # ==========================================
        # Information
        # ==========================================                                                    
        st.info("⚠ Upload only H&E stained oral histopathological images. Clinical photographs or mobile camera images are not supported.")                                                      
        st.warning("This application is intended for research and educational purposes only and should not be used as the sole basis for clinical diagnosis.")
        st.success("✅ ATPF Segmentation Completed Successfully")