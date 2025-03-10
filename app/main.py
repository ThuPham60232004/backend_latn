from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import threading
from app.services.image_preprocessing import process_new_images
from app.services.mmrag import mainclient 

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Tự động quét và xử lý ảnh khi server khởi động"""
    thread1 = threading.Thread(target=process_new_images)
    thread1.start()

    thread2 = threading.Thread(target=mainclient) 
    thread2.start()
@app.get("/")
async def root():
    return {"message": "Image Processing API is running!"}
