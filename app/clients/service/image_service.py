import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Load model CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_image(image_path):
    """Tiền xử lý ảnh: Gaussian Blur, cân bằng histogram, phát hiện cạnh."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Làm mờ bằng Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Cân bằng histogram
    equalized = cv2.equalizeHist(blurred)

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(equalized, 50, 150)

    return edges
def extract_image_features(image_path):
    """Tiền xử lý ảnh & Trích xuất vector đặc trưng từ ảnh bằng CLIP."""
    # Tiền xử lý ảnh trước
    processed_image = preprocess_image(image_path)

    # Chuyển đổi sang ảnh PIL (RGB)
    processed_image = Image.fromarray(processed_image).convert("RGB")

    # Trích xuất đặc trưng bằng CLIP
    inputs = processor(images=processed_image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return features.squeeze().numpy()

