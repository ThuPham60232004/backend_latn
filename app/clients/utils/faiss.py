import faiss
import numpy as np
import config
import os

# Load FAISS index nếu tồn tại, nếu không thì tạo mới
def load_or_create_faiss():
    if os.path.exists(config.FAISS_INDEX_PATH):
        return faiss.read_index(config.FAISS_INDEX_PATH)
    return faiss.IndexFlatL2(512)  # Vector 512 chiều

index = load_or_create_faiss()

def add_vector_to_faiss(vector):
    """Thêm vector mới vào FAISS index."""
    index.add(np.array([vector]).astype("float32"))
    faiss.write_index(index, config.FAISS_INDEX_PATH)

