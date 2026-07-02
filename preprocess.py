import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

test_transform = A.Compose([
    A.Resize(512, 512),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


def preprocess_image(image):

    image = np.array(image.convert("RGB"))

    transformed = test_transform(image=image)

    image = transformed["image"]

    image = image.unsqueeze(0)

    return image