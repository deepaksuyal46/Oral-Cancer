import os
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from predict import DEVICE, predict_image


st.set_page_config(
    page_title="ATPFv2 Oral Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root{
    --bg:#eef3f7;
    --panel:#fbfcfe;
    --panel-tint:#f2f7f8;
    --ink:#17212b;
    --muted:#627184;
    --line:#cfd9e3;
    --brand:#116a73;
    --brand-dark:#0d3f4a;
    --brand-soft:#dff1f2;
    --accent:#c77b35;
    --accent-soft:#fff2e7;
    --success:#0f766e;
    --warning:#b45309;
    --danger:#b91c1c;
    --shadow:0 18px 48px rgba(32,49,67,.10);
}

.stApp{
    background:
        linear-gradient(135deg, rgba(17,106,115,.10), rgba(199,123,53,.06) 42%, rgba(238,243,247,0) 72%),
        radial-gradient(circle at top right, rgba(17,106,115,.14), transparent 32rem),
        var(--bg);
    color:var(--ink);
}

.block-container{
    max-width:1320px;
    padding-top:2rem;
    padding-bottom:2rem;
}

[data-testid="stSidebar"]{
    background:
        linear-gradient(180deg, #092f38 0%, #122633 48%, #171f2a 100%);
    border-right:1px solid rgba(255,255,255,.10);
}

[data-testid="stSidebar"] *{
    color:#f9fafb;
}

[data-testid="stSidebar"] .stAlert *{
    color:#111827;
}

.app-header{
    position:relative;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.26);
    background:
        linear-gradient(135deg, rgba(8,48,57,.96), rgba(17,106,115,.92) 54%, rgba(50,73,87,.94)),
        linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,.08) 1px, transparent 1px);
    background-size:auto, 34px 34px, 34px 34px;
    border-radius:8px;
    padding:34px 34px;
    margin-bottom:24px;
    box-shadow:var(--shadow);
}

.app-header:after{
    content:"";
    position:absolute;
    right:28px;
    top:26px;
    width:148px;
    height:148px;
    border:1px solid rgba(255,255,255,.24);
    border-radius:50%;
    opacity:.55;
}

.header-content{
    position:relative;
    z-index:1;
    max-width:900px;
}

.header-meta{
    display:flex;
    flex-wrap:wrap;
    gap:10px;
    margin-top:20px;
}

.header-chip{
    border:1px solid rgba(255,255,255,.24);
    background:rgba(255,255,255,.10);
    color:#e6fffb;
    border-radius:999px;
    font-size:12px;
    font-weight:700;
    padding:7px 11px;
}

.eyebrow{
    color:#f8c99d;
    font-size:12px;
    font-weight:700;
    letter-spacing:.08em;
    text-transform:uppercase;
    margin-bottom:8px;
}

.app-title{
    color:#ffffff;
    font-size:38px;
    line-height:1.18;
    font-weight:760;
    margin:0 0 10px;
}

.app-subtitle{
    color:#d7e6e8;
    font-size:16px;
    line-height:1.55;
    max-width:860px;
    margin:0;
}

.panel{
    border:1px solid var(--line);
    background:linear-gradient(180deg, var(--panel), var(--panel-tint));
    border-radius:8px;
    padding:20px;
    margin-bottom:18px;
    box-shadow:0 12px 32px rgba(36,54,70,.07);
    border-top:3px solid rgba(17,106,115,.55);
}

.panel-title{
    color:var(--ink);
    font-size:18px;
    font-weight:720;
    margin:0 0 4px;
}

.panel-caption{
    color:var(--muted);
    font-size:14px;
    margin:0 0 16px;
}

.side-brand{
    border:1px solid rgba(255,255,255,.14);
    background:rgba(255,255,255,.07);
    border-radius:8px;
    padding:16px;
    margin-bottom:14px;
}

.side-brand h1{
    color:#ffffff;
    font-size:24px;
    margin:0 0 6px;
}

.side-brand p{
    color:#cbd5e1;
    font-size:13px;
    margin:0;
}

.side-stat{
    border:1px solid rgba(255,255,255,.14);
    background:linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.045));
    border-radius:8px;
    padding:13px 14px;
    margin-bottom:10px;
}

.side-stat span{
    color:#cbd5e1;
    display:block;
    font-size:12px;
    margin-bottom:4px;
}

.side-stat strong{
    color:#ffffff;
    display:block;
    font-size:18px;
}

.status-pill{
    border:1px solid rgba(17,106,115,.25);
    background:var(--brand-soft);
    color:#0d3f4a;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    gap:8px;
    font-size:13px;
    font-weight:650;
    padding:7px 11px;
}

.summary-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px;
    margin:8px 0 18px;
}

.summary-card{
    border:1px solid rgba(207,217,227,.92);
    background:linear-gradient(180deg,#ffffff,#f7fafb);
    border-radius:8px;
    padding:16px;
    box-shadow:0 12px 26px rgba(36,54,70,.06);
    position:relative;
    overflow:hidden;
}

.summary-card:before{
    content:"";
    position:absolute;
    left:0;
    top:0;
    width:4px;
    height:100%;
    background:linear-gradient(180deg, var(--brand), var(--accent));
}

.summary-card span{
    display:block;
    color:var(--muted);
    font-size:12px;
    font-weight:650;
    margin-bottom:8px;
    text-transform:uppercase;
}

.summary-card strong{
    color:var(--ink);
    display:block;
    font-size:24px;
    line-height:1.1;
}

.summary-card small{
    color:var(--muted);
    display:block;
    font-size:12px;
    margin-top:8px;
}

.result-note{
    border-left:4px solid var(--brand);
    background:linear-gradient(90deg, #edfafa, #fffaf5);
    padding:13px 15px;
    border-radius:6px;
    color:var(--muted);
    font-size:14px;
    line-height:1.5;
}

.image-title{
    color:var(--ink);
    font-size:15px;
    font-weight:700;
    margin:0 0 10px;
}

.download-row{
    border:1px solid var(--line);
    border-radius:8px;
    background:linear-gradient(180deg,#ffffff,#f6faf9);
    padding:14px;
    margin-bottom:12px;
    color:var(--muted);
    font-weight:650;
}

[data-testid="stFileUploader"]{
    border:1px dashed rgba(17,106,115,.45);
    background:#f6fbfb;
    border-radius:8px;
    padding:10px;
}

[data-testid="stImage"] img{
    border:1px solid var(--line);
    border-radius:8px;
    box-shadow:0 14px 34px rgba(36,54,70,.12);
}

[data-testid="metric-container"]{
    border:1px solid var(--line);
    background:linear-gradient(180deg,#ffffff,#f7fafb);
    border-radius:8px;
    padding:14px 16px;
}

.stButton>button,
.stDownloadButton>button{
    border-radius:6px;
    border:1px solid var(--brand);
    background:linear-gradient(180deg, #16828c, var(--brand));
    color:#ffffff;
    font-weight:700;
    min-height:42px;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
    border-color:var(--brand-dark);
    background:linear-gradient(180deg, #116a73, var(--brand-dark));
    color:#ffffff;
}

.stTabs [data-baseweb="tab-list"]{
    gap:6px;
}

.stTabs [data-baseweb="tab"]{
    border-radius:6px 6px 0 0;
    padding:10px 14px;
    background:#f5f8f9;
    border:1px solid var(--line);
}

@media (max-width: 900px){
    .summary-grid{
        grid-template-columns:repeat(2,minmax(0,1fr));
    }
    .app-title{
        font-size:28px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def encode_png(image_array):
    success, buffer = cv2.imencode(".png", image_array)
    if not success:
        raise ValueError("Could not encode image as PNG.")
    return buffer.tobytes()


def build_report(result):
    return f"""ATPFv2 Oral Cancer Histopathology Analysis
=========================================
Source              : {result["source_name"]}
Model               : ATPFv2
Encoder             : ResNet34
Device              : {result["device"]}
Confidence          : {result["confidence"] * 100:.2f} %
Segmented Area      : {result["area"]:.2f} %
Inference Time      : {result["inference_time"]:.3f} sec
Input Size          : 512 x 512
Generated By        : ATPFv2 Oral Cancer AI Platform
=========================================
Research and educational use only.
This output must not be used as the sole basis for diagnosis.
"""


def confidence_label(confidence):
    if confidence > 0.90:
        return "High confidence", "success"
    if confidence > 0.70:
        return "Moderate confidence", "warning"
    return "Low confidence", "error"


def run_analysis(image, source_name):
    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Reading image",
        "Preprocessing tissue patch",
        "Extracting tissue features",
        "Running ATPFv2 segmentation",
        "Preparing visual outputs",
    ]

    for index, step in enumerate(steps, start=1):
        status.info(step)
        progress.progress(index / len(steps))
        time.sleep(0.2)

    start = time.time()
    probability, mask, confidence, area = predict_image(image)
    inference_time = time.time() - start

    progress.empty()
    status.empty()

    original = np.array(image)
    original = cv2.resize(original, (512, 512))

    heatmap = cv2.applyColorMap(probability, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = original.copy()
    red = np.zeros_like(original)
    red[:, :, 2] = 255
    overlay[mask > 0] = (0.62 * original[mask > 0] + 0.38 * red[mask > 0]).astype(np.uint8)

    st.session_state.analysis_counter = st.session_state.get("analysis_counter", 0) + 1
    st.session_state.analysis_result = {
        "id": st.session_state.analysis_counter,
        "source_name": source_name,
        "original": original,
        "overlay": overlay,
        "heatmap": heatmap,
        "mask": mask,
        "confidence": confidence,
        "area": area,
        "inference_time": inference_time,
        "device": DEVICE.type.upper(),
    }


def render_sidebar():
    st.sidebar.markdown(
        """
<div class="side-brand">
    <h1>ATPFv2 AI</h1>
    <p>Oral histopathology segmentation workspace</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
<div class="side-stat"><span>Model</span><strong>ATPFv2</strong></div>
<div class="side-stat"><span>Encoder</span><strong>ResNet34</strong></div>
<div class="side-stat"><span>Device</span><strong>{DEVICE.type.upper()}</strong></div>
<div class="side-stat"><span>Input</span><strong>512 x 512 RGB</strong></div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Workflow")
    st.sidebar.markdown(
        """
1. Select an upload or demo image
2. Review the image preview
3. Run ATPFv2 analysis
4. Inspect segmentation outputs
5. Download masks and report
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Research use only. This tool assists histopathology segmentation and must not replace expert pathological diagnosis."
    )


def render_header():
    st.markdown(
        """
<section class="app-header">
    <div class="header-content">
        <div class="eyebrow">Research segmentation platform</div>
        <h1 class="app-title">ATPFv2 Oral Cancer Histopathological AI Platform</h1>
        <p class="app-subtitle">
            Upload or select an H&E stained tissue patch, run ATPFv2 inference, inspect the segmentation output,
            and export reproducible mask, overlay, and report files.
        </p>
        <div class="header-meta">
            <span class="header-chip">ATPFv2 model</span>
            <span class="header-chip">ResNet34 encoder</span>
            <span class="header-chip">512 x 512 RGB input</span>
            <span class="header-chip">Research use only</span>
        </div>
    </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_input_panel():
    st.markdown(
        """
<div class="panel">
    <div class="panel-title">Input Image</div>
    <p class="panel-caption">Use H&E stained oral histopathological images. Clinical photographs are not supported.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Image source",
        ["Upload image", "Try demo image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None
    source_name = "No image selected"

    input_col, preview_col = st.columns([0.92, 1.08], gap="large")

    with input_col:
        if source_mode == "Upload image":
            uploaded_file = st.file_uploader(
                "Upload PNG, JPG, JPEG, or TIF",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                source_name = uploaded_file.name
        else:
            demo_files = sorted(os.listdir("sample_images")) if os.path.isdir("sample_images") else []
            if demo_files:
                demo_image = st.selectbox("Choose demo image", demo_files)
                image = Image.open(os.path.join("sample_images", demo_image)).convert("RGB")
                source_name = demo_image
            else:
                st.warning("No demo images were found in the sample_images folder.")

        analyze = st.button(
            "Run ATPFv2 Analysis",
            type="primary",
            use_container_width=True,
            disabled=image is None,
        )

    with preview_col:
        if image is not None:
            st.markdown(f'<span class="status-pill">Ready: {source_name}</span>', unsafe_allow_html=True)
            st.image(image, caption=source_name, use_container_width=True)
        else:
            st.info("Select an image to enable analysis.")

    if analyze and image is not None:
        run_analysis(image, source_name)


def render_results():
    result = st.session_state.get("analysis_result")
    if not result:
        st.markdown(
            """
<div class="panel">
    <div class="panel-title">Analysis Results</div>
    <p class="panel-caption">Results will appear here after you run ATPFv2 analysis.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    label, level = confidence_label(result["confidence"])
    st.markdown(
        f"""
<div class="panel">
    <div class="panel-title">Analysis Results</div>
    <p class="panel-caption">Latest result for <strong>{result["source_name"]}</strong></p>
</div>
<div class="summary-grid">
    <div class="summary-card"><span>Confidence</span><strong>{result["confidence"] * 100:.2f}%</strong><small>{label}</small></div>
    <div class="summary-card"><span>Segmented Area</span><strong>{result["area"]:.2f}%</strong><small>Mask coverage</small></div>
    <div class="summary-card"><span>Inference</span><strong>{result["inference_time"]:.3f}s</strong><small>Model runtime</small></div>
    <div class="summary-card"><span>Device</span><strong>{result["device"]}</strong><small>Inference engine</small></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if level == "success":
        st.success("High confidence prediction. Review the overlay and mask before exporting.")
    elif level == "warning":
        st.warning("Moderate confidence prediction. Review the visual outputs carefully.")
    else:
        st.error("Low confidence prediction. Treat the output as uncertain and verify manually.")

    overview_tab, images_tab, downloads_tab = st.tabs(["Overview", "Visual Results", "Downloads"])

    with overview_tab:
        st.markdown(
            """
<div class="result-note">
    ATPFv2 highlights regions selected by the segmentation model. The binary mask is thresholded from the
    probability map and the overlay blends detected pixels onto the resized 512 x 512 input image.
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Model", "ATPFv2")
        st.metric("Encoder", "ResNet34")

    with images_tab:
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown('<p class="image-title">Original Image</p>', unsafe_allow_html=True)
            st.image(result["original"], use_container_width=True)
        with row1_col2:
            st.markdown('<p class="image-title">Segmentation Overlay</p>', unsafe_allow_html=True)
            st.image(result["overlay"], use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown('<p class="image-title">Probability Heatmap</p>', unsafe_allow_html=True)
            st.image(result["heatmap"], use_container_width=True)
        with row2_col2:
            st.markdown('<p class="image-title">Binary Mask</p>', unsafe_allow_html=True)
            st.image(result["mask"], use_container_width=True)

    with downloads_tab:
        analysis_id = result["id"]
        overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="download-row">Binary segmentation mask</div>', unsafe_allow_html=True)
            st.download_button(
                "Download Mask",
                data=encode_png(result["mask"]),
                file_name="ATPFv2_Binary_Mask.png",
                mime="image/png",
                key=f"mask_download_{analysis_id}",
                use_container_width=True,
            )
        with col2:
            st.markdown('<div class="download-row">Overlay visualization</div>', unsafe_allow_html=True)
            st.download_button(
                "Download Overlay",
                data=encode_png(overlay_bgr),
                file_name="ATPFv2_Overlay.png",
                mime="image/png",
                key=f"overlay_download_{analysis_id}",
                use_container_width=True,
            )
        with col3:
            st.markdown('<div class="download-row">Plain-text prediction report</div>', unsafe_allow_html=True)
            st.download_button(
                "Download Report",
                data=build_report(result),
                file_name="ATPFv2_Prediction_Report.txt",
                mime="text/plain",
                key=f"report_download_{analysis_id}",
                use_container_width=True,
            )

    st.info("Upload only H&E stained oral histopathological images. Clinical photographs or mobile camera images are not supported.")
    st.warning("Research and educational use only. Do not use this app as the sole basis for clinical diagnosis.")


render_sidebar()
render_header()
render_input_panel()
render_results()
