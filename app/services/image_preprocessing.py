import cv2
import os
import numpy as np
from pathlib import Path

# Thư mục chứa dữ liệu ảnh gốc và thư mục chứa ảnh sau xử lý
IMAGE_DIR = "app/static/images/test_set"
PROCESSED_DIR = "app/static/processed"

# Đảm bảo thư mục lưu ảnh đã xử lý tồn tại
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_all_images(directory):
    """Lấy tất cả các ảnh (JPG, JPEG, PNG) trong thư mục và các thư mục con"""
    return list(Path(directory).rglob("*.jpg")) + \
           list(Path(directory).rglob("*.jpeg")) + \
           list(Path(directory).rglob("*.png"))

def is_new_image(image_path):
    """Kiểm tra xem ảnh đã được xử lý hay chưa bằng cách so sánh với thư mục processed"""
    processed_image_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
    return not processed_image_path.exists()

def preprocess_image(image_path):
    """Tiền xử lý ảnh: Gaussian Blur, cân bằng histogram, phát hiện cạnh"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return None

    # Áp dụng Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Cân bằng histogram
    equalized = cv2.equalizeHist(blurred)
    # Phát hiện cạnh bằng Canny
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
            # Xác định đường dẫn lưu ảnh sau xử lý (giữ nguyên cấu trúc thư mục con)
            output_path = Path(PROCESSED_DIR) / Path(image_path).relative_to(IMAGE_DIR)
            os.makedirs(output_path.parent, exist_ok=True)  # Tạo thư mục nếu chưa có
            cv2.imwrite(str(output_path), processed)
            print(f"Đã xử lý: {image_path}")

    print("Hoàn thành xử lý ảnh mới.")

