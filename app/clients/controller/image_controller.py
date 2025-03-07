from services.image_service import extract_image_features
from utils.storage import upload_to_gcs
from utils.faiss_index import add_vector_to_faiss, search_similar_images

def process_and_store_image(image_path):
    """Xử lý ảnh, lưu vào GCS và thêm vào FAISS."""
    vector = extract_image_features(image_path)
    gcs_url = upload_to_gcs(image_path, image_path.split("/")[-1])
    add_vector_to_faiss(vector)
    return {"message": "Image processed", "url": gcs_url}


