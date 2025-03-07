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
from concurrent.futures import ThreadPoolExecutor
import asyncio
import hashlib
load_dotenv()

IMAGE_DIR = "app/static/images/test_set"
PROCESSED_DIR = "app/static/processed"
INDEX_FILE = "app/static/faiss_index.bin"
PROCESSED_IMAGES_FILE = "app/static/processed_images.json"
EMBEDDED_VECTORS_FILE = "app/static/embedded_vectors.json"

GCS_BUCKET = "lantn"
GCS_INDEX_PATH = "faiss_index.bin"
GCS_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
    raise FileNotFoundError("Google Cloud Key không tồn tại hoặc chưa được cấu hình đúng.")
INDEX_DIM = 512  
index = faiss.IndexFlatL2(INDEX_DIM)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_all_images(directory):
    """Lấy tất cả các ảnh (JPG, JPEG, PNG) trong thư mục và các thư mục con"""
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def load_processed_images():
    """Tải danh sách ảnh đã xử lý từ JSON."""
    if os.path.exists(PROCESSED_IMAGES_FILE):
        try:
            with open(PROCESSED_IMAGES_FILE, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            return set()
    return set()

def save_processed_images(processed_images):
    """Lưu danh sách ảnh đã xử lý vào JSON."""
    with open(PROCESSED_IMAGES_FILE, "w") as f:
        json.dump(list(processed_images), f)

def preprocess_image(image_path):
    """Tiền xử lý ảnh: Gaussian Blur, cân bằng histogram, phát hiện cạnh"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def load_index():
    """Tải FAISS Index từ file nếu có."""
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return faiss.IndexFlatL2(INDEX_DIM)

index = load_index()

def save_index():
    """Lưu FAISS Index vào file."""
    faiss.write_index(index, INDEX_FILE)

def load_processed_vectors():
    """Tải danh sách ảnh và vector nhúng từ JSON."""
    if os.path.exists(EMBEDDED_VECTORS_FILE):
        try:
            with open(EMBEDDED_VECTORS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        except json.JSONDecodeError:
            return {}
    return {}

def save_processed_vectors(processed_vectors):
    """Lưu danh sách ảnh và vector nhúng vào JSON."""
    with open(EMBEDDED_VECTORS_FILE, "w") as f:
        json.dump({k: (np.array(v, dtype=np.float32).tolist() if isinstance(v, list) else v.tolist()) for k, v in processed_vectors.items()}, f)

def embed_image(image_path):
    """Nhúng ảnh thành vector sử dụng CLIP."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def process_new_images():
    """Xử lý ảnh mới và nhúng ảnh thành vector nếu chưa có."""
    all_images = get_all_images(IMAGE_DIR)
    processed_vectors = load_processed_vectors()
    new_images = [str(img) for img in all_images if str(img) not in processed_vectors]

    print(f"Tổng số ảnh: {len(all_images)}, Ảnh mới cần nhúng: {len(new_images)}")

    if not new_images:
        print("Không có ảnh mới để xử lý.")
        return
    
    for image_path in new_images:
        processed = preprocess_image(image_path)
        if processed is not None:
            output_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
            os.makedirs(output_path.parent, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            
            embedding = embed_image(image_path)
            if embedding is not None and embedding.shape[1] == INDEX_DIM:
                index.add(embedding)
                processed_vectors[image_path] = embedding.flatten().tolist()
                print(f"Đã nhúng vector: {image_path}")
            else:
                print(f"Lỗi nhúng vector: {image_path}, shape: {embedding.shape if embedding is not None else None}")

    if index.ntotal > 0:
        save_index()
        print("Đã lưu FAISS Index.")

    save_processed_vectors(processed_vectors)
    print("Hoàn thành xử lý và nhúng ảnh mới.")
def get_file_checksum(file_path):
    """Tính checksum của file để kiểm tra thay đổi."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Upload file lên Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    prev_checksum = get_file_checksum(local_path)
    
    blob.upload_from_filename(local_path)
    print(f"Đã upload FAISS Index lên GCS: {gcs_path}")

    new_checksum = get_file_checksum(local_path)
    if prev_checksum == new_checksum:
        print("Không có thay đổi, không cần upload lại.")

async def async_upload_to_gcs(local_path, bucket_name, gcs_path):
    """Upload file lên GCS bất đồng bộ."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, upload_to_gcs, local_path, bucket_name, gcs_path)

if __name__ == "__main__":
    process_new_images()
    asyncio.run(async_upload_to_gcs(INDEX_FILE, GCS_BUCKET, GCS_INDEX_PATH))


