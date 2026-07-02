import os
import gdown
import torch
import numpy as np

from model import UNetATPF
from preprocess import preprocess_image

# =====================================================
# Device Configuration
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Model Path
# =====================================================

MODEL_PATH = "saved_models/ATPFv2_Fold2.pth"

# =====================================================
# Download Model Automatically (First Time Only)
# =====================================================

if not os.path.exists(MODEL_PATH):

    print("Downloading trained model...")

    os.makedirs("saved_models", exist_ok=True)

    FILE_ID = "136yj6desfWlf9LqaRRv544XNDIIZQC6v"

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

    print("Model downloaded successfully.")

# =====================================================
# Load Trained Model
# =====================================================

model = UNetATPF()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# =====================================================
# Prediction Function
# =====================================================

def predict_image(image, threshold=0.5):
    """
    Predict segmentation mask from a PIL image.

    Returns:
        probability_map : uint8 image (0-255)
        binary_mask     : uint8 image (0 or 255)
        confidence      : float (0-1)
        segmented_area  : float (%)
    """

    # -----------------------------
    # Preprocess
    # -----------------------------

    image = preprocess_image(image)
    image = image.to(DEVICE)

    # -----------------------------
    # Inference
    # -----------------------------

    with torch.no_grad():

        masks, extras = model(image)

        probability = torch.sigmoid(masks)

        binary = (probability > threshold).float()

    # -----------------------------
    # Convert to NumPy
    # -----------------------------

    probability = probability.squeeze().cpu().numpy()
    binary = binary.squeeze().cpu().numpy()

    # -----------------------------
    # Confidence
    # -----------------------------

    if np.sum(binary) > 0:
        confidence = float(probability[binary > 0].mean())
    else:
        confidence = float(probability.mean())

    # -----------------------------
    # Segmented Area
    # -----------------------------

    segmented_area = float(
        ((binary > 0).sum() / binary.size) * 100
    )

    # -----------------------------
    # Convert Images
    # -----------------------------

    probability_map = (probability * 255).astype(np.uint8)
    binary_mask = (binary * 255).astype(np.uint8)

    return (
        probability_map,
        binary_mask,
        confidence,
        segmented_area
    )