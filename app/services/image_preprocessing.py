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
EMBEDDED_VECTORS_FILE = "app/static/embedded_vectors.json"
LABELS_FILE = "app/static/labels.npy"

# Google Cloud Storage
GCS_BUCKET = "lantn"
GCS_INDEX_PATH = "faiss_index.bin"
GCS_LABELS_PATH = "labels.npy"

# FAISS Index
INDEX_DIM = 512  
index = faiss.IndexFlatL2(INDEX_DIM)

# Kiểm tra thiết bị GPU hay CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_all_images(directory):
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def extract_label_from_filename(filename):
    try:
        label = filename.split("_")[1].split("-")[1].split(" ")[0]
        return label.lower()
    except IndexError:
        return "unknown"

def load_json(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Lỗi đọc file JSON: {file_path}")
    return {}

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Lỗi đọc ảnh: {image_path}")
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def load_index():
    if os.path.exists(INDEX_FILE):
        try:
            return faiss.read_index(INDEX_FILE)
        except Exception as e:
            print(f"⚠️ Lỗi tải FAISS Index: {e}, tạo Index mới.")
    return faiss.IndexFlatL2(INDEX_DIM)

index = load_index()

def save_index():
    faiss.write_index(index, INDEX_FILE)

def embed_image(image_path):
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

def upload_to_gcs(local_path, gcs_path):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)

        if os.path.exists(local_path):
            blob.upload_from_filename(local_path)
            print(f"✅ Đã upload {local_path} lên GCS tại: gs://{GCS_BUCKET}/{gcs_path}")
        else:
            print(f"❌ Không tìm thấy file {local_path} để upload!")
    except Exception as e:
        print(f"❌ Lỗi upload {local_path} lên GCS: {e}")

def process_new_images():
    all_images = get_all_images(IMAGE_DIR)
    processed_vectors = load_json(EMBEDDED_VECTORS_FILE)
    labels_dict = {}

    try:
        if os.path.exists(LABELS_FILE):
            labels_dict = np.load(LABELS_FILE, allow_pickle=True).item()
    except Exception as e:
        print(f"⚠️ Lỗi tải labels.npy: {e}")

    new_images = [str(img) for img in all_images if str(img) not in processed_vectors]

    print(f"📂 Tổng số ảnh: {len(all_images)}, Ảnh mới cần nhúng: {len(new_images)}")

    if not new_images:
        print("✅ Không có ảnh mới để xử lý.")
        return
    
    for i, image_path in enumerate(new_images):
        label = extract_label_from_filename(Path(image_path).name)

        processed = preprocess_image(image_path)
        if processed is not None:
            output_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
            os.makedirs(output_path.parent, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            
            embedding = embed_image(image_path)
            if embedding is not None:
                index.add(embedding)
                processed_vectors[image_path] = embedding.flatten().tolist()
                labels_dict[index.ntotal - 1] = label
                print(f"✅ Đã nhúng vector: {image_path} - {label}")
            else:
                print(f"❌ Lỗi nhúng vector: {image_path}")

    if index.ntotal > 0:
        save_index()
        print("💾 Đã lưu FAISS Index.")

    save_json(processed_vectors, EMBEDDED_VECTORS_FILE)
    np.save(LABELS_FILE, labels_dict)
    print("✅ Đã lưu nhãn bệnh vào labels.npy")

    upload_to_gcs(INDEX_FILE, GCS_INDEX_PATH)
    upload_to_gcs(LABELS_FILE, GCS_LABELS_PATH)

if __name__ == "__main__":
    process_new_images()
