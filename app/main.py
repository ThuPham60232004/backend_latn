from fastapi import FastAPI
from app.services.image_preprocessing import process_new_images
import threading

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Tự động quét và xử lý ảnh từ tất cả thư mục con khi server khởi động"""
    thread = threading.Thread(target=process_new_images)
    thread.start()

@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}
