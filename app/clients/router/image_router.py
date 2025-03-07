from fastapi import APIRouter, UploadFile, File
import shutil
from controllers.image_controller import process_and_store_image

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload ảnh, xử lý và lưu trữ."""
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    response = process_and_store_image(file_path)
    return response


