import cv2
import os
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Định nghĩa các đường dẫn
IMAGE_DIR = "app/static/images/test_set"
PROCESSED_DIR = "app/static/processed"
INDEX_FILE = "app/static/faiss_index.bin"
PROCESSED_IMAGES_FILE = "app/static/processed_images.json"
EMBEDDED_VECTORS_FILE = "app/static/embedded_vectors.json"

# Google Cloud Storage
GCS_BUCKET = "lantn"
GCS_INDEX_PATH = "faiss_index.bin"
GCS_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
    raise FileNotFoundError("❌ Google Cloud Key không tồn tại hoặc chưa được cấu hình đúng!")

# FAISS Index
INDEX_DIM = 512  
index = faiss.IndexFlatL2(INDEX_DIM)

# Kiểm tra thiết bị GPU hay CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Tạo thư mục nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_all_images(directory):
    """Lấy danh sách ảnh JPG, JPEG, PNG từ thư mục."""
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def load_json(file_path):
    """Tải dữ liệu từ JSON, trả về set nếu lỗi."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Lỗi đọc file JSON: {file_path}")
    return {}

def save_json(data, file_path):
    """Lưu dữ liệu vào JSON."""
    with open(file_path, "w") as f:
        json.dump(data, f)

def preprocess_image(image_path):
    """Tiền xử lý ảnh: Gaussian Blur, cân bằng histogram, phát hiện cạnh."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Lỗi đọc ảnh: {image_path}")
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def load_index():
    """Tải FAISS Index từ file nếu có."""
    if os.path.exists(INDEX_FILE):
        try:
            return faiss.read_index(INDEX_FILE)
        except Exception as e:
            print(f"⚠️ Lỗi tải FAISS Index: {e}, tạo Index mới.")
    return faiss.IndexFlatL2(INDEX_DIM)

index = load_index()

def save_index():
    """Lưu FAISS Index vào file."""
    faiss.write_index(index, INDEX_FILE)

def embed_image(image_path):
    """Nhúng ảnh thành vector sử dụng CLIP."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Lỗi đọc ảnh: {image_path}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    
    embedding = embedding.cpu().numpy().astype(np.float32)

    if embedding.shape[1] != INDEX_DIM:
        print(f"⚠️ Vector nhúng có shape không đúng: {embedding.shape}")
        return None
    
    return embedding

def upload_faiss_to_gcs():
    """Upload FAISS Index lên Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_INDEX_PATH)

        if os.path.exists(INDEX_FILE):
            blob.upload_from_filename(INDEX_FILE)
            print(f"✅ FAISS Index đã được upload lên GCS tại: gs://{GCS_BUCKET}/{GCS_INDEX_PATH}")
        else:
            print("❌ Không tìm thấy FAISS Index để upload!")
    except Exception as e:
        print(f"❌ Lỗi upload FAISS Index lên GCS: {e}")

def process_new_images():
    """Xử lý ảnh mới và nhúng ảnh thành vector nếu chưa có."""
    all_images = get_all_images(IMAGE_DIR)
    processed_vectors = load_json(EMBEDDED_VECTORS_FILE)
    new_images = [str(img) for img in all_images if str(img) not in processed_vectors]

    print(f"📂 Tổng số ảnh: {len(all_images)}, Ảnh mới cần nhúng: {len(new_images)}")

    if not new_images:
        print("✅ Không có ảnh mới để xử lý.")
        return
    
    for image_path in new_images:
        processed = preprocess_image(image_path)
        if processed is not None:
            output_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
            os.makedirs(output_path.parent, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            
            embedding = embed_image(image_path)
            if embedding is not None:
                index.add(embedding)
                processed_vectors[image_path] = embedding.flatten().tolist()
                print(f"✅ Đã nhúng vector: {image_path}")
            else:
                print(f"❌ Lỗi nhúng vector: {image_path}")

    if index.ntotal > 0:
        save_index()
        print("💾 Đã lưu FAISS Index.")

    save_json(processed_vectors, EMBEDDED_VECTORS_FILE)
    print("✅ Hoàn thành xử lý và nhúng ảnh mới.")
    
    upload_faiss_to_gcs()

if __name__ == "__main__":
    process_new_images()
