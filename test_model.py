import torch
import traceback

path = "saved_models/ATPFv2_Fold2.pth"

try:
    checkpoint = torch.load(path, map_location="cpu")
    print("Loaded successfully!")
    print(type(checkpoint))

except Exception:
    traceback.print_exc()