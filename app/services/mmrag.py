import cv2
import os
import numpy as np
import torch
import faiss
import json
from pathlib import Path
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GCS_BUCKET = "lantn"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
    raise FileNotFoundError("Google Cloud Key không tồn tại hoặc chưa được cấu hình đúng!")
INDEX_FILE = "app/static/faiss_index.bin"
VECTOR_FILE = "static/processed/embedded_vectors.json"
LABELS_FILE = "app/static/labels.npy"
INDEX_DIM = 512  

index = None
labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def upload_to_gcs(local_path, destination_blob_name):
    """Upload file lên Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"✅ Đã upload {local_path} lên GCS tại: gs://{GCS_BUCKET}/{destination_blob_name}")

def preprocess_image(image_path):
    """Tiền xử lý ảnh bằng Gaussian Blur và Canny Edge Detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def embed_image(image_path):
    """Nhúng ảnh thành vector sử dụng mô hình CLIP."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def search_similar_images(query_vector, top_k=5):
    """Tìm ảnh tương tự bằng FAISS Index."""
    if index is None or index.ntotal == 0:
        print("❌ FAISS index trống!")
        return []

    distances, indices = index.search(query_vector, top_k)
    print(f"🔍 Chỉ số tìm thấy: {indices}")

    # Đảm bảo index không vượt phạm vi nhãn
    similar_labels = []
    for i in indices[0]:
        if 0 <= i < len(labels):
            similar_labels.append(labels[i])
        else:
            print(f"⚠️ Index {i} vượt phạm vi labels ({len(labels)})!")
            similar_labels.append("unknown")

    return similar_labels

def load_faiss_index():
    """Tải FAISS Index và nhãn bệnh từ file."""
    global index, labels
    if os.path.exists(INDEX_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            print(f"✅ FAISS Index tải thành công! Tổng số vector: {index.ntotal}")
        except Exception as e:
            print(f"❌ Lỗi tải FAISS Index: {e}")
            index = None
    else:
        print("❌ FAISS Index không tồn tại!")
    if os.path.exists(LABELS_FILE):
        labels = np.load(LABELS_FILE, allow_pickle=True).tolist()
        print(f"✅ Đã tải {len(labels)} nhãn bệnh từ labels.npy")
    else:
        print("❌ labels.npy không tồn tại!")

def get_disease_description(disease_name):
    """Dùng Gemini AI để lấy mô tả chi tiết về bệnh."""
    prompt = f"""
    Tôi cần mô tả chi tiết về bệnh {disease_name}, bao gồm nguyên nhân, triệu chứng, cách điều trị và phòng ngừa.
    Trình bày ngắn gọn, dễ hiểu.
    """

    try:
        response = genai.GenerativeModel("gemini-1.5-pro-002").generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi gọi Gemini API: {e}")
        return "Không có thông tin chi tiết."

def process_image(image_path):
    """Xử lý ảnh đầu vào và tìm kiếm nhãn bệnh."""
    processed = preprocess_image(image_path)
    if processed is not None:
        processed_path = f"processed_{Path(image_path).name}"
        cv2.imwrite(processed_path, processed)
        upload_to_gcs(processed_path, GCS_IMAGE_PATH + processed_path)

        embedding = embed_image(image_path)
        if embedding is not None:
            result_labels = search_similar_images(embedding)
            if len(set(result_labels)) > 1:
                print("🤖 Dùng AI Agent để xác nhận nhãn bệnh...")
                best_label = verify_with_gemini(result_labels)
            else:
                best_label = result_labels[0] if result_labels else "unknown"
            description = get_disease_description(best_label) if best_label != "unknown" else "Không xác định được bệnh."
            return {"disease": best_label, "description": description}

    return {"disease": "unknown", "description": "Không xác định được bệnh."}

def mainclient():
    """Hàm chính để kiểm tra hoạt động của hệ thống."""
    load_faiss_index()

    # Nếu FAISS Index chưa tải được, tạo mới
    if index is None or index.ntotal == 0:
        create_faiss_index()

    # Kiểm tra số lượng nhãn có khớp với FAISS Index không
    if index and len(labels) != index.ntotal:
        print(f"⚠️ Cảnh báo: FAISS Index ({index.ntotal}) và labels ({len(labels)}) không khớp!")

    image_path = "photo/chickenpox/1_VI-chickenpox (12).jpg"
    print("📂 File tồn tại:", os.path.exists(image_path))
    
    # Tìm kiếm ảnh tương tự
    result = process_image(image_path)
    print(f"🩺 Kết quả bệnh: {result['disease']}")
    print(f"📄 Mô tả bệnh:\n{result['description']}")

if __name__ == "__main__":
    mainclient()
