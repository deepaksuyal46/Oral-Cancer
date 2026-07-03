import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from predict import DEVICE, predict_image

st.set_page_config(
    page_title="Oral Cancer Histopathology Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS matching the Reference UI perfectly
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #f8fafc;
    --card-bg: #ffffff;
    --card-soft: #f8fafc;
    --border: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --brand-blue: #3b82f6;
    --brand-light: #eff6ff;
    --brand-dark: #1e293b;
    --success: #10b981;
    --sidebar-bg: #1e293b;
    --sidebar-active: #3b82f6;
    --sidebar-text: #f8fafc;
    --shadow-card: 0 4px 14px rgba(15, 23, 42, 0.07);
    --hero-start: #e0f2fe;
    --hero-end: #ede9fe;
    --hero-border: #bae6fd;
    --button-text: #ffffff;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: #0b1220;
        --card-bg: #111827;
        --card-soft: #172033;
        --border: #263449;
        --text-main: #f8fafc;
        --text-muted: #cbd5e1;
        --brand-blue: #60a5fa;
        --brand-light: #172554;
        --brand-dark: #f8fafc;
        --success: #34d399;
        --sidebar-bg: #020617;
        --sidebar-active: #60a5fa;
        --sidebar-text: #f8fafc;
        --shadow-card: 0 10px 28px rgba(0, 0, 0, 0.35);
        --hero-start: #0f2a44;
        --hero-end: #241c45;
        --hero-border: #1e3a5f;
        --button-text: #06101f;
    }
}

/* Global resets */
.stApp {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Top Padding reduction */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: var(--sidebar-text) !important;
}

/* Custom Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: var(--shadow-card);
    height: 100%;
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
}

.card-title i {
    color: var(--brand-blue);
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, var(--hero-start) 0%, var(--hero-end) 100%);
    border-radius: 16px;
    padding: 32px;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
    border: 1px solid var(--hero-border);
}
.hero-content {
    position: relative;
    z-index: 2;
    max-width: 600px;
}
.hero-badge {
    background: rgba(255,255,255,0.12);
    color: var(--brand-blue);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 12px;
    border: 1px solid rgba(59,130,246,0.2);
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--brand-dark);
    margin: 0 0 12px 0;
}
.hero-title span {
    color: var(--brand-blue);
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 24px;
}
.feature-pills {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.feature-pill {
    background: var(--card-bg);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}
.feature-pill span {
    color: #8b5cf6;
}
.hero-image {
    position: absolute;
    right: 20px;
    bottom: -20px;
    height: 120%;
    opacity: 0.9;
    z-index: 1;
}

/* Stepper */
.stepper-container {
    display: flex;
    justify-content: space-between;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.step {
    display: flex;
    align-items: center;
    gap: 12px;
}
.step-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--brand-light);
    color: var(--brand-blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
}
.step-text {
    display: flex;
    flex-direction: column;
}
.step-text strong {
    font-size: 13px;
    color: var(--brand-dark);
}
.step-text span {
    font-size: 11px;
    color: var(--text-muted);
}
.step-arrow {
    color: #cbd5e1;
    font-size: 14px;
}

/* Upload Area specific */
.upload-card {
    text-align: center;
    padding: 40px 20px;
    border: 2px dashed #93c5fd;
    border-radius: 12px;
    background: var(--card-soft);
    margin-top: 10px;
    transition: all 0.3s ease;
}
.upload-card:hover {
    border-color: var(--brand-blue);
    background: var(--brand-light);
}
.upload-icon {
    font-size: 32px;
    color: var(--brand-blue);
    margin-bottom: 12px;
}

/* AI Summary Circular Chart */
.circular-chart {
    display: block;
    margin: 0 auto;
    max-width: 120px;
    max-height: 120px;
}
.circle-bg {
    fill: none;
    stroke: #e2e8f0;
    stroke-width: 3;
}
.circle {
    fill: none;
    stroke-width: 3;
    stroke-linecap: round;
    transition: stroke-dasharray 1.5s ease-out;
}
.percentage {
    fill: var(--text-main);
    font-family: 'Inter', sans-serif;
    font-size: 0.6em;
    text-anchor: middle;
    font-weight: 700;
}
.conf-label {
    fill: var(--success);
    font-family: 'Inter', sans-serif;
    font-size: 0.25em;
    text-anchor: middle;
    font-weight: 600;
}

.stats-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.stat-box {
    background: var(--card-soft);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-icon {
    background: #d1fae5;
    color: #059669;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}
.stat-text strong {
    display: block;
    font-size: 16px;
    color: var(--text-main);
}
.stat-text span {
    font-size: 11px;
    color: var(--text-muted);
}

.interpretation-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px;
    margin-top: 16px;
}
.interpretation-box strong {
    font-size: 12px;
    color: #b45309;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.interpretation-box p {
    font-size: 11px;
    color: #78350f;
    margin: 0;
    line-height: 1.4;
}

/* Tabs overriding */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding-top: 10px;
    padding-bottom: 10px;
    color: var(--text-muted);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: var(--brand-blue) !important;
    border-bottom: 2px solid var(--brand-blue) !important;
}

/* Bottom Images */
.image-box {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--card-bg);
}
.image-box-header {
    background: var(--card-soft);
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    justify-content: space-between;
}
.image-box-header i {
    color: var(--brand-blue);
    cursor: pointer;
}

/* Right Sidebar "Understanding the Results" */
.understand-item {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.u-icon {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.u-icon.heatmap { background: #fee2e2; color: #ef4444; }
.u-icon.mask { background: #d1fae5; color: #10b981; }
.u-icon.overlay { background: #e0e7ff; color: #4f46e5; }

.u-text strong {
    font-size: 13px;
    color: var(--text-main);
    display: block;
    margin-bottom: 2px;
}
.u-text p {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.4;
}

/* Hide default file uploader styling */
div[data-testid="stFileUploader"] section {
    padding: 10px;
}
.stButton>button {
    background: var(--brand-blue);
    color: var(--button-text);
    border-radius: 8px;
    border: none;
    font-weight: 600;
    width: 100%;
}
.stButton>button:hover {
    background: #2563eb;
    color: #ffffff;
}

[data-testid="stSidebar"] .stButton>button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-left: 4px solid transparent !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    justify-content: flex-start !important;
    margin-bottom: 8px !important;
    min-height: 46px !important;
    padding-left: 14px !important;
    text-align: left !important;
    transition: all .22s ease !important;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(59,130,246,0.24) !important;
    border-color: rgba(147,197,253,0.42) !important;
    border-left-color: #60a5fa !important;
    color: #ffffff !important;
    transform: translateX(3px);
}
.sidebar-section-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    margin: 22px 0 10px;
    text-transform: uppercase;
}
.sidebar-tip {
    background: rgba(59,130,246,.12);
    border: 1px solid rgba(147,197,253,.20);
    border-radius: 10px;
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.45;
    margin-top: 14px;
    padding: 12px;
}
.view-heading {
    align-items: center;
    background: linear-gradient(90deg, var(--brand-light), var(--card-bg));
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 16px;
    padding: 14px 16px;
}
.view-heading strong {
    color: var(--text-main);
    font-size: 15px;
}
.view-heading span {
    color: var(--text-muted);
    font-size: 12px;
}
.download-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 8px 18px rgba(15,23,42,.05);
    margin-bottom: 12px;
    padding: 14px;
}
.download-card strong {
    color: var(--text-main);
    display: block;
    font-size: 13px;
    margin-bottom: 4px;
}
.download-card p {
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1.4;
    margin: 0;
}

.understanding-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow-card);
    padding: 18px;
}
.understanding-intro {
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.55;
    margin: 0 0 16px;
}
.explain-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}
.explain-card {
    background: var(--card-soft);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.explain-card:hover {
    border-color: var(--brand-blue);
    box-shadow: var(--shadow-card);
    transform: translateY(-2px);
}
.explain-title {
    align-items: center;
    color: var(--text-main);
    display: flex;
    font-size: 15px;
    font-weight: 700;
    gap: 8px;
    margin: 0 0 9px;
}
.info-chip {
    align-items: center;
    background: var(--brand-light);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--brand-blue);
    display: inline-flex;
    font-size: 11px;
    font-weight: 800;
    height: 20px;
    justify-content: center;
    width: 20px;
}
.explain-card p {
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.55;
    margin: 0;
}
.legend-list {
    display: grid;
    gap: 8px;
    margin-top: 12px;
}
.legend-row {
    align-items: center;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-muted);
    display: flex;
    font-size: 12px;
    gap: 9px;
    padding: 7px 10px;
}
.legend-dot {
    border: 1px solid rgba(15,23,42,.18);
    border-radius: 50%;
    flex: 0 0 13px;
    height: 13px;
    width: 13px;
}
.dot-blue { background: #2563eb; }
.dot-green { background: #22c55e; }
.dot-yellow { background: #facc15; }
.dot-red { background: #ef4444; }
.dot-white { background: #ffffff; }
.dot-black { background: #111827; }
.confidence-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 12px;
}
.confidence-box {
    border-radius: 12px;
    padding: 13px;
    border: 1px solid var(--border);
}
.confidence-box strong {
    color: var(--text-main);
    display: block;
    font-size: 13px;
    margin-bottom: 6px;
}
.confidence-box span {
    color: var(--text-muted);
    display: block;
    font-size: 12px;
    line-height: 1.45;
}
.confidence-high { background: rgba(16,185,129,.12); }
.confidence-mid { background: rgba(245,158,11,.14); }
.confidence-low { background: rgba(239,68,68,.12); }
.medical-disclaimer {
    background: linear-gradient(90deg, var(--brand-light), var(--card-bg));
    border: 1px solid var(--border);
    border-left: 4px solid var(--brand-blue);
    border-radius: 14px;
    margin-top: 14px;
    padding: 16px;
}
.medical-disclaimer strong {
    color: var(--text-main);
    display: block;
    font-size: 14px;
    margin-bottom: 8px;
}
.medical-disclaimer p {
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.55;
    margin: 0;
}

.card p,
.card li,
.card .stMarkdown,
.card label {
    color: var(--text-muted);
}

.card h1,
.card h2,
.card h3,
.card h4,
.card strong {
    color: var(--text-main);
}

@media (max-width: 900px) {
    .hero-banner {
        padding: 24px;
    }
    .hero-title {
        font-size: 26px;
    }
    .stepper-container {
        display: grid;
        gap: 12px;
    }
    .step-arrow {
        display: none;
    }
    .explain-grid,
    .confidence-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


def encode_png(image_array):
    success, buffer = cv2.imencode(".png", image_array)
    if not success:
        raise ValueError("Could not encode image as PNG.")
    return buffer.tobytes()

def build_report(result):
    return f"""Oral Cancer Histopathological Image Assistant - AI Review Report
================================================
Image Type          : H&E-stained histopathological microscopic tissue image
Result Strength     : {result["confidence"] * 100:.2f} %
Highlighted Area    : {result["area"]:.2f} %
Analysis Time       : {result["time"]:.3f} sec

Interpretation:
The highlighted regions are AI-generated predictions from the uploaded histopathological tissue image.
They may require closer expert review by a qualified pathologist.
This output is for research and education only.

Important:
This software must not be used as the sole basis for diagnosis or treatment decisions.
Always interpret results with a qualified pathologist and clinical findings.
"""

def render_understanding_section():
    with st.expander("📖 Understanding Your AI Analysis", expanded=False):
        st.markdown(
            """
<div class="understanding-card">
    <p class="understanding-intro">
        This guide explains each image and number in simple language so doctors, researchers,
        students, and non-technical users can understand the AI result.
    </p>
    <div class="explain-grid">
        <div class="explain-card">
            <div class="explain-title">🖼 Original Histopathological Image <span class="info-chip">i</span></div>
            <p>
                This is the original H&amp;E-stained microscopic tissue image uploaded by the user.
                It serves as the input for AI analysis and represents the tissue before any processing.
            </p>
        </div>
        <div class="explain-card">
            <div class="explain-title">🌡 Probability Heatmap <span class="info-chip">i</span></div>
            <p>
                The heatmap visualizes how confident the AI model is across different tissue regions.
                Brighter colors indicate areas where the model predicts a higher likelihood of abnormal tissue.
            </p>
            <div class="legend-list">
                <div class="legend-row"><span class="legend-dot dot-blue"></span>Blue → Very Low Probability</div>
                <div class="legend-row"><span class="legend-dot dot-green"></span>Green → Low to Moderate Probability</div>
                <div class="legend-row"><span class="legend-dot dot-yellow"></span>Yellow → Moderate Probability</div>
                <div class="legend-row"><span class="legend-dot dot-red"></span>Red → High Probability of suspicious tissue</div>
            </div>
        </div>
        <div class="explain-card">
            <div class="explain-title">🧩 Binary Segmentation Mask <span class="info-chip">i</span></div>
            <p>
                This is the final prediction produced by the AI. The mask is obtained after applying
                the model's decision threshold to the probability map.
            </p>
            <div class="legend-list">
                <div class="legend-row"><span class="legend-dot dot-white"></span>White → Regions identified as suspicious tissue</div>
                <div class="legend-row"><span class="legend-dot dot-black"></span>Black → Background or normal tissue</div>
            </div>
        </div>
        <div class="explain-card">
            <div class="explain-title">🎨 Segmentation Overlay <span class="info-chip">i</span></div>
            <p>
                The overlay combines the original microscopic image with the AI-generated segmentation.
                Red highlighted regions represent tissue areas detected by the AI as suspicious.
                This visualization allows users to compare the original tissue with the predicted abnormal regions.
            </p>
        </div>
        <div class="explain-card">
            <div class="explain-title">📊 AI Prediction Summary <span class="info-chip">i</span></div>
            <p>
                <strong>Confidence</strong> indicates how confident the AI model is in its segmentation prediction.<br>
                <strong>Segmented Area</strong> represents the percentage of the histopathological image predicted as suspicious tissue.<br>
                <strong>Inference Time</strong> indicates the time taken by the AI model to complete the analysis.
            </p>
        </div>
        <div class="explain-card">
            <div class="explain-title">🧭 Interpretation Guide <span class="info-chip">i</span></div>
            <div class="confidence-grid">
                <div class="confidence-box confidence-high">
                    <strong>🟢 High Confidence (&gt;90%)</strong>
                    <span>The segmentation is highly consistent according to the AI model.</span>
                </div>
                <div class="confidence-box confidence-mid">
                    <strong>🟡 Moderate Confidence (75–90%)</strong>
                    <span>The prediction is reasonably reliable, but visual review is recommended.</span>
                </div>
                <div class="confidence-box confidence-low">
                    <strong>🔴 Lower Confidence (&lt;75%)</strong>
                    <span>The prediction should be interpreted carefully and validated by an expert.</span>
                </div>
            </div>
        </div>
    </div>
    <div class="medical-disclaimer">
        <strong>Research Use Only</strong>
        <p>
            This application provides AI-assisted tissue segmentation for research and educational purposes.
            The highlighted regions are algorithmic predictions and should always be interpreted by a qualified
            pathologist together with clinical findings. This software must not be used as the sole basis for
            diagnosis or treatment decisions.
        </p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

def render_sidebar():
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 32px;">
        <div style="background:#3b82f6; width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:18px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v20M14 2v20M2 10h20M2 14h20M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/></svg>
        </div>
        <div>
            <h3 style="margin:0; font-size:16px; font-weight:700;">Oral Cancer</h3>
            <span style="font-size:12px; color:#94a3b8;">Histopathology Assistant</span>
        </div>
    </div>
    <div class="sidebar-section-title">Navigation</div>
    """, unsafe_allow_html=True)

    if "result_view" not in st.session_state:
        st.session_state.result_view = "Overview"

    if st.sidebar.button("🖼 Image Review", key="nav_image", use_container_width=True):
        st.session_state.result_view = "Overview"
    if st.sidebar.button("🌡 Color Map Guide", key="nav_heatmap", use_container_width=True):
        st.session_state.result_view = "Heatmap"
    if st.sidebar.button("🧩 Marked Area View", key="nav_mask", use_container_width=True):
        st.session_state.result_view = "Segmentation Mask"
    if st.sidebar.button("🎨 Overlay View", key="nav_overlay", use_container_width=True):
        st.session_state.result_view = "Overlay"
    if st.sidebar.button("📖 Understanding Guide", key="nav_guide", use_container_width=True):
        st.session_state.result_view = "Information"
    if st.sidebar.button("📄 Reports & Downloads", key="nav_reports", use_container_width=True):
        st.session_state.result_view = "Downloads"
    if st.sidebar.button("ℹ About This App", key="nav_about", use_container_width=True):
        st.session_state.result_view = "About"

    st.sidebar.markdown(
        f"""
<div class="sidebar-tip">
    Current view: <strong>{st.session_state.result_view}</strong><br>
    These buttons control the result panel after analysis.
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Model")
    st.sidebar.markdown("**ATPFv2**")
    st.sidebar.caption("Deep Learning Model")

    st.sidebar.caption("System Status")
    st.sidebar.success("Online - all systems operational")
    st.sidebar.info("Need help? Open the Understanding Guide from the navigation buttons.")


def render_topbar_and_hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-content">
            <div class="hero-badge">Simple Image Review</div>
            <h1 class="hero-title">Oral Cancer <span>Histopathology Assistant</span></h1>
            <p class="hero-subtitle">Upload an H&amp;E-stained histopathological microscopic tissue image, review the highlighted regions, and read a clear explanation of what each result means.</p>
            <div class="feature-pills">
                <div class="feature-pill"><span>🖼</span> Easy comparison</div>
                <div class="feature-pill"><span>📖</span> Clear guide</div>
                <div class="feature-pill"><span>📄</span> Download report</div>
                <div class="feature-pill"><span>🔬</span> Research use</div>
            </div>
        </div>
    </div>
    
    <div class="stepper-container">
        <div class="step">
            <div class="step-icon" style="background:#3b82f6; color:white;">☁️</div>
            <div class="step-text">
                <strong>1 Upload Histopathological Image</strong>
                <span>Select a tissue slide image</span>
            </div>
        </div>
        <div class="step-arrow">›</div>
        <div class="step">
            <div class="step-icon">✨</div>
            <div class="step-text">
                <strong>2 Tissue Image Check</strong>
                <span>App reviews tissue patterns</span>
            </div>
        </div>
        <div class="step-arrow">›</div>
        <div class="step">
            <div class="step-icon">👁️</div>
            <div class="step-text">
                <strong>3 Review Results</strong>
                <span>Compare highlighted areas</span>
            </div>
        </div>
        <div class="step-arrow">›</div>
        <div class="step">
            <div class="step-icon">📥</div>
            <div class="step-text">
                <strong>4 Download Report</strong>
                <span>Download results</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_middle_section():
    col1, col2, col3 = st.columns([1.2, 2.2, 1])
    
    image = None
    source_name = "No histopathological image"
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">📤 Upload Histopathological Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Histopathological Image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        
        st.markdown("""
        <div style="text-align:center; font-size:12px; color:#64748b; margin-top:8px;">or</div>
        """, unsafe_allow_html=True)
        
        demo_files = sorted(os.listdir("sample_images")) if os.path.isdir("sample_images") else []
        if demo_files:
            demo_image = st.selectbox("Use Sample Image", demo_files, label_visibility="collapsed")
        
        analyze_btn = st.button("Check Histopathological Image")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            source_name = uploaded_file.name
        elif demo_files:
            image = Image.open(os.path.join("sample_images", demo_image)).convert("RGB")
            source_name = demo_image

    with col2:
        st.markdown('<div class="card"><div class="card-title">🖼️ Histopathological Image Preview</div>', unsafe_allow_html=True)
        if image is not None:
            st.image(image, use_container_width=True)
            st.caption(f"Selected histopathological image: {source_name}")
        else:
            st.markdown('<div style="height:300px; display:flex; align-items:center; justify-content:center; background:var(--card-soft); border:1px dashed var(--border); border-radius:8px; color:var(--text-muted);">No histopathological image selected</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        if "analysis_result" in st.session_state and analyze_btn:
            pass # Use session state if already analyzed, but since we re-analyze on button click:
            
        if analyze_btn and image is not None:
            with st.spinner("Processing..."):
                start_time = time.time()
                probability, mask, confidence, area = predict_image(image)
                inference_time = time.time() - start_time
                
                original = np.array(image)
                original = cv2.resize(original, (512, 512))
                heatmap = cv2.applyColorMap(probability, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = original.copy()
                red = np.zeros_like(original)
                red[:, :, 2] = 255
                overlay[mask > 0] = (0.62 * original[mask > 0] + 0.38 * red[mask > 0]).astype(np.uint8)

                st.session_state.analysis_result = {
                    "original": original,
                    "heatmap": heatmap,
                    "mask": mask,
                    "overlay": overlay,
                    "confidence": confidence,
                    "area": area,
                    "time": inference_time
                }
        
        result = st.session_state.get("analysis_result")
        
        st.markdown('<div class="card"><div class="card-title">⏱️ Histopathological Image Check Summary</div>', unsafe_allow_html=True)
        
        if result:
            conf = result["confidence"] * 100
            area = result["area"]
            inf_time = result["time"]
            stroke_dash = f"{conf}, 100"
            color = "#10b981" if conf > 90 else "#f59e0b"
            
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-top:20px;">
                <div style="flex:1;">
                    <svg viewBox="0 0 36 36" class="circular-chart">
                        <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                        <path class="circle" stroke="{color}" stroke-dasharray="{stroke_dash}" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                        <text x="18" y="18" class="percentage">{conf:.1f}%</text>
                        <text x="18" y="24" class="conf-label">Strength</text>
                    </svg>
                </div>
                <div class="stats-grid" style="flex:1;">
                    <div class="stat-box">
                        <div class="stat-icon" style="background:#d1fae5; color:#059669;">✨</div>
                        <div class="stat-text"><strong>{area:.1f}%</strong><span>Highlighted Area</span></div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-icon" style="background:#e0f2fe; color:#0284c7;">⏱️</div>
                        <div class="stat-text"><strong>{inf_time:.3f}s</strong><span>Time Taken</span></div>
                    </div>
                </div>
            </div>
            
            <div class="interpretation-box">
                <strong>ℹ️ Interpretation</strong>
                <p>Regions highlighted in this histopathological image may require closer expert review. This is not a diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="height:200px; display:flex; align-items:center; justify-content:center; color:#94a3b8; font-size:13px;">Run analysis to see summary</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)


def render_bottom_section():
    result = st.session_state.get("analysis_result")
    if not result:
        if st.session_state.get("result_view") in {"Heatmap", "Segmentation Mask", "Overlay", "Information", "Downloads", "About"}:
            st.warning("Run AI Analysis first. The selected section will appear here after a result is available.")
        return
        
    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

    view_options = ["Overview", "Heatmap", "Segmentation Mask", "Overlay", "Information", "Downloads", "About"]
    if st.session_state.get("result_view") not in view_options:
        st.session_state.result_view = "Overview"

    selected_view = st.radio(
        "Result section",
        view_options,
        horizontal=True,
        key="result_view",
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="view-heading">
            <strong>{selected_view}</strong>
            <span>Use the left sidebar buttons or this selector to switch sections.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if selected_view == "Overview":
        img_c1, img_c2, img_c3, img_c4 = st.columns(4)
        with img_c1:
            st.markdown('<div class="image-box"><div class="image-box-header">Original Histopathological Image</div>', unsafe_allow_html=True)
            st.image(result["original"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with img_c2:
            st.markdown('<div class="image-box"><div class="image-box-header">Heatmap</div>', unsafe_allow_html=True)
            st.image(result["heatmap"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with img_c3:
            st.markdown('<div class="image-box"><div class="image-box-header">Segmentation Mask</div>', unsafe_allow_html=True)
            st.image(result["mask"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with img_c4:
            st.markdown('<div class="image-box"><div class="image-box-header">Overlay Result</div>', unsafe_allow_html=True)
            st.image(result["overlay"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        render_understanding_section()

    elif selected_view == "Heatmap":
        left, right = st.columns([1.4, 1])
        with left:
            st.image(result["heatmap"], caption="Probability Heatmap", use_container_width=True)
        with right:
            st.markdown("""
            ### 🔥 Probability Heatmap
            The heatmap shows where the AI model is more or less confident.

            - 🔵 Blue: very low probability
            - 🟢 Green: low to moderate probability
            - 🟡 Yellow: moderate probability
            - 🔴 Red: high probability of suspicious tissue
            """)

    elif selected_view == "Segmentation Mask":
        left, right = st.columns([1.4, 1])
        with left:
            st.image(result["mask"], caption="Binary Segmentation Mask", use_container_width=True)
        with right:
            st.markdown("""
            ### 🧩 Segmentation Mask
            This is the AI's final marked-area prediction.

            - ⚪ White: suspicious tissue region
            - ⚫ Black: background or normal tissue

            The mask is created after applying the model's decision threshold.
            """)

    elif selected_view == "Overlay":
        left, right = st.columns([1.4, 1])
        with left:
            st.image(result["overlay"], caption="Overlay Result", use_container_width=True)
        with right:
            st.markdown("""
            ### 🎨 Overlay Result
            The overlay places the AI prediction on top of the original microscopic image.

            🔴 Red highlighted regions represent tissue areas detected by the AI as suspicious.
            This is usually the easiest view for doctors, students, and non-technical users to understand.
            """)

    elif selected_view == "Information":
        render_understanding_section()

    elif selected_view == "Downloads":
        overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="download-card"><strong>Marked Area</strong><p>Download the black-white AI mask.</p></div>', unsafe_allow_html=True)
            st.download_button("Download Marked Area", encode_png(result["mask"]), "Marked_Area.png", "image/png", use_container_width=True, key="download_mask")
        with col2:
            st.markdown('<div class="download-card"><strong>Highlighted Histopathological Image</strong><p>Download the original histopathological image with red AI highlights.</p></div>', unsafe_allow_html=True)
            st.download_button("Download Highlighted Histopathological Image", encode_png(overlay_bgr), "Highlighted_Histopathological_Image.png", "image/png", use_container_width=True, key="download_overlay")
        with col3:
            st.markdown('<div class="download-card"><strong>Report</strong><p>Download a simple text summary.</p></div>', unsafe_allow_html=True)
            st.download_button("Download Report", build_report(result), "Image_Check_Report.txt", "text/plain", use_container_width=True, key="download_report")

    elif selected_view == "About":
        st.markdown("""
        ### ℹ About This App

        Oral Cancer Histopathology Assistant is an AI-assisted research tool for reviewing histopathological microscopic tissue images.
        It helps highlight regions that may deserve closer expert review.

        This application is designed for doctors, researchers, students, and non-technical users, but final
        interpretation must always be performed by qualified professionals.
        """)

    st.markdown('</div>', unsafe_allow_html=True)

render_sidebar()
render_topbar_and_hero()
render_middle_section()
render_bottom_section()
