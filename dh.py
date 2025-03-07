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

# Load biến môi trường từ file .env
load_dotenv()

# Định nghĩa đường dẫn
IMAGE_DIR = "app/static/images/test_set"
PROCESSED_DIR = "app/static/processed"
INDEX_FILE = "app/static/faiss_index.bin"
PROCESSED_IMAGES_FILE = "app/static/processed_images.json"

# Cấu hình Google Cloud Storage (GCS)
GCS_BUCKET = "lantn"
GCS_INDEX_PATH = "faiss_index.bin"
GCS_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Kiểm tra key GCS
if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
    raise FileNotFoundError("Google Cloud Key không tồn tại hoặc chưa được cấu hình đúng.")

# Khởi tạo FAISS Index
INDEX_DIM = 512  # Kích thước vector của CLIP ViT-B/32
index = faiss.IndexFlatL2(INDEX_DIM)

# Load model CLIP từ transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Đảm bảo thư mục lưu ảnh đã xử lý tồn tại
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_all_images(directory):
    """Lấy tất cả các ảnh (JPG, JPEG, PNG) trong thư mục và các thư mục con"""
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def load_processed_images():
    """Tải danh sách ảnh đã nhúng từ file JSON."""
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_images(processed_images):
    """Lưu danh sách ảnh đã nhúng vào file JSON."""
    with open(PROCESSED_IMAGES_FILE, "w") as f:
        json.dump(list(processed_images), f)

def is_new_image(image_path):
    """Kiểm tra xem ảnh đã được xử lý hay chưa"""
    processed_image_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
    return not processed_image_path.exists()

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

def process_new_images():
    """Quét toàn bộ thư mục và xử lý ảnh mới"""
    all_images = get_all_images(IMAGE_DIR)
    new_images = [img for img in all_images if is_new_image(img)]
    
    print(f"Tổng số ảnh tìm thấy: {len(all_images)}")
    print(f"Số ảnh mới cần xử lý: {len(new_images)}")

    if not new_images:
        print("Không có ảnh mới để xử lý.")
        return

    for image_path in new_images:
        processed = preprocess_image(str(image_path))
        if processed is not None:
            output_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
            os.makedirs(output_path.parent, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            print(f"Đã xử lý: {image_path}")

    print("Hoàn thành xử lý ảnh mới.")

def load_image(image_path):
    """Đọc ảnh và tiền xử lý cho CLIP"""
    print ('Đọc ảnh và tiền xử lý cho CLIP')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển về RGB
    inputs = processor(images=img, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

def get_clip_embedding(image_path):
    """Nhúng ảnh thành vector bằng CLIP từ transformers"""
    print(f"🔄 Đang xử lý ảnh: {image_path}")
    inputs = load_image(image_path)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    vector = image_features.cpu().numpy().astype(np.float32)
    print(f"✅ Vector có kích thước: {vector.shape}")
    print(f"📌 Vector đầu ra (một phần): {vector.flatten()[:10]}") 
    if vector.shape != (1, 512): 
        print(f"Lỗi kích thước vector cho ảnh {image_path}: {vector.shape}")
    
    return vector

def add_to_faiss(image_vectors):
    """Thêm vector vào FAISS Index"""
    print(f"Thêm vector vào FAISS, kích thước: {image_vectors.shape}")
    
    if len(image_vectors.shape) == 1:  
        image_vectors = np.expand_dims(image_vectors, axis=0)

    index.add(image_vectors)
    print(f"FAISS hiện có {index.ntotal} vectors.")

def save_faiss_index():
    """Lưu FAISS Index xuống file"""
    faiss.write_index(index, INDEX_FILE)

def load_faiss_index():
    """Tải FAISS Index từ file (nếu có)"""
    global index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)

def upload_faiss_to_gcs():
    """Tải FAISS Index lên Google Cloud Storage"""
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_INDEX_PATH)
    blob.upload_from_filename(INDEX_FILE)
    print(f"Đã tải FAISS Index lên GCS: {GCS_INDEX_PATH}")

def download_faiss_from_gcs():
    """Tải FAISS Index từ Google Cloud Storage nếu có."""
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_INDEX_PATH)

    if blob.exists():
        blob.download_to_filename(INDEX_FILE)
        print(f"Đã tải FAISS Index từ GCS về: {INDEX_FILE}")
        load_faiss_index()
    else:
        print("Không tìm thấy FAISS Index trên GCS, tạo mới.")

def process_images_for_embedding():
    """Duyệt toàn bộ ảnh đã xử lý, nhúng thành vector và lưu vào FAISS"""
    all_images = get_all_images(PROCESSED_DIR)  # Chỉ lấy ảnh đã xử lý
    processed_images = load_processed_images()
    new_images = [img for img in all_images if str(img) not in processed_images]

    print(f"🔍 Tổng số ảnh đã xử lý: {len(all_images)}")
    print(f"🆕 Số ảnh mới cần nhúng: {len(new_images)}")

    if not new_images:
        print("⚠️ Không có ảnh mới cần nhúng.")
        return

    for img_path in new_images:
        print(f"🔄 Đang nhúng ảnh: {img_path}")
        vector = get_clip_embedding(str(img_path))

        if vector is None:
            print(f"❌ Lỗi khi nhúng ảnh {img_path}, bỏ qua.")
            continue

        print(f"✅ Vector của {img_path.name}: {vector.shape}")

        # add_to_faiss(vector)  # Thêm vào FAISS Index
        processed_images.add(str(img_path))  # Lưu lại ảnh đã nhúng
        print(f"✔️ Đã nhúng: {img_path.name}")

    # save_faiss_index()
    # save_processed_images(processed_images)
    # upload_faiss_to_gcs()
    test_folder = "app/static/processed/BA-impetigo"  # Thư mục chứa ảnh
    image_files = list(Path(test_folder).glob("*.png"))  # Lấy danh sách ảnh PNG
    if os.path.exists(test_folder):
        print(f"✅ Thư mục {test_folder} tồn tại.")
    else:
        print(f"❌ Thư mục {test_folder} không tồn tại! Kiểm tra lại đường dẫn.")
        print(f"📂 Số ảnh tìm thấy trong thư mục: {len(image_files)}")

    for img_path in image_files:
        print(f"\n🔄 Đang xử lý ảnh: {img_path}")
        vector = get_clip_embedding(str(img_path))

        if vector is not None:
            print(f"✅ Vector đầu ra có kích thước: {vector.shape}")
            print(f"📌 Một phần vector: {vector.flatten()[:10]}")
        else:
            print(f"❌ Lỗi: Không tạo được vector cho ảnh {img_path}")


if __name__ == "__main__":
    # download_faiss_from_gcs()  # Tải FAISS từ GCS về nếu có
    process_new_images()  # Tiền xử lý ảnh mới
    process_images_for_embedding()  # Nhúng vector và lưu lên FAISS
#  sau khi quét thư mục để lấy tập dữ liệu có sẵn,tiền xử lý ảnh hãy làm thêm chức năng Nhúng ảnh thành vector → Thêm vào FAISS → Lưu FAISS → Upload lên Google Cloud Storage