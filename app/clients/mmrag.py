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

    # Load biến môi trường
    load_dotenv()

    # Cấu hình Google Cloud Storage
    GCS_BUCKET = "lantn"
    GCS_IMAGE_PATH = "uploaded_images/"
    GCS_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
        raise FileNotFoundError("❌ Google Cloud Key không tồn tại hoặc chưa được cấu hình đúng!")

    # FAISS Index
    INDEX_FILE = "app/static/faiss_index.bin"
    VECTOR_FILE = "static/processed/embedded_vectors.json"
    INDEX_DIM = 512  
    index = None

    # Kiểm tra thiết bị GPU hay CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load mô hình CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Kết nối Google Cloud Storage
    def upload_to_gcs(local_path, destination_blob_name):
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        print(f"✅ Đã upload {local_path} lên GCS tại: gs://{GCS_BUCKET}/{destination_blob_name}")

    # Tiền xử lý ảnh
    def preprocess_image(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Lỗi đọc ảnh: {image_path}")
            return None

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        edges = cv2.Canny(equalized, 50, 150)
        return edges

    # Nhúng ảnh thành vector
    def embed_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Lỗi đọc ảnh: {image_path}")
            return None
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        
        return embedding.cpu().numpy().astype(np.float32)

    # Tìm kiếm ảnh tương tự
    def search_similar_images(query_vector, top_k=5):
        if index is None or index.ntotal == 0:
            print("⚠️ FAISS Index rỗng!")
            return []
        
        distances, indices = index.search(query_vector, top_k)
        return indices[0].tolist()

    # Tải FAISS Index
    def load_faiss_index():
        global index
        if os.path.exists(INDEX_FILE):
            try:
                index = faiss.read_index(INDEX_FILE)
                print(f"✅ FAISS Index tải thành công! Tổng số vector: {index.ntotal}")
            except Exception as e:
                print(f"❌ Lỗi khi tải FAISS Index: {e}")
                index = None
        else:
            print("❌ FAISS Index không tồn tại!")

    # Tạo FAISS Index từ dữ liệu JSON nếu chưa có
    def create_faiss_index():
        global index
        if os.path.exists(VECTOR_FILE):
            with open(VECTOR_FILE, "r") as f:
                data = json.load(f)
            
            if data:
                vectors = np.array(data["vectors"], dtype=np.float32)
                index = faiss.IndexFlatL2(vectors.shape[1])
                index.add(vectors)
                faiss.write_index(index, INDEX_FILE)
                print(f"✅ FAISS Index được tạo với {index.ntotal} vectors!")
            else:
                print("⚠️ embedded_vectors.json không có dữ liệu!")
        else:
            print("❌ Không tìm thấy embedded_vectors.json!")

    # Xử lý ảnh đầu vào
    def process_image(image_path):
        processed = preprocess_image(image_path)
        if processed is not None:
            processed_path = f"processed_{Path(image_path).name}"
            cv2.imwrite(processed_path, processed)
            
            # Upload ảnh xử lý lên GCS
            upload_to_gcs(processed_path, GCS_IMAGE_PATH + processed_path)
            
            # Nhúng ảnh và tìm kiếm
            embedding = embed_image(image_path)
            if embedding is not None:
                similar_images = search_similar_images(embedding)
                return similar_images
        
        return []

    # Khởi chạy
    if __name__ == "__main__":
        load_faiss_index()  # Tải FAISS Index
        if index is None or index.ntotal == 0:
            create_faiss_index()  # Tạo FAISS Index nếu cần

        image_path = "D:/backend_latn/photo/chickenpox/1_VI-chickenpox (12).jpg"  # Ảnh đầu vào
        result = process_image(image_path)
        print("🔍 Kết quả tìm kiếm ảnh tương tự:", result)
