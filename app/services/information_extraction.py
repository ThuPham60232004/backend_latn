import docx
import fitz
import re
import os
from docx import Document

# Hàm đọc file PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Hàm đọc file DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Hàm đọc file TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

# Hàm chính để xác định loại file và đọc nội dung
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        return "Không hỗ trợ đọc file này."

# Chuẩn hóa văn bản
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Xóa khoảng trắng thừa
    text = re.sub(r"\n+", "\n", text)  # Xóa dòng trắng
    sentences = re.split(r"[.!?]\s+", text)  # Tách câu
    return sentences

# Trích xuất thông tin quan trọng bằng Regex
def extract_medical_info(text):
    """Trích xuất thông tin y khoa bằng Regex"""
    sentences = clean_text(text)  # Dùng re.split thay cho sent_tokenize

    extracted_info = {
        "Tên bệnh": None,
        "Triệu chứng": None,
        "Chẩn đoán": None,
        "Điều trị": None,
        "Phòng bệnh": None
    }

    patterns = {
        "Tên bệnh": r"(Bệnh|Hội chứng)\s([\w\s]+)",
        "Triệu chứng": r"(Triệu chứng|Biểu hiện):\s([\w\s,]+)",
        "Chẩn đoán": r"(Chẩn đoán):\s([\w\s,]+)",
        "Điều trị": r"(Điều trị):\s([\w\s,]+)",
        "Phòng bệnh": r"(Phòng bệnh|Phòng ngừa):\s([\w\s,]+)"
    }

    for key, pattern in patterns.items():
        for sentence in sentences:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                extracted_info[key] = match.group(2)
                break

    return extracted_info

# Kiểm tra dữ liệu bị thiếu
def check_missing_data(extracted_info):
    """Xác định các mục bị thiếu dữ liệu"""
    missing_sections = [key for key, value in extracted_info.items() if not value]
    return missing_sections

# Xử lý nhiều file
def process_multiple_files(directory):
    """Xử lý tất cả các tài liệu trong thư mục"""
    files = [f for f in os.listdir(directory) if f.endswith(('.pdf', '.docx', '.txt'))]
    
    if not files:
        print("❌ Không có tài liệu nào trong thư mục!")
        return
    
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"\n📂 Đang xử lý file: {file}\n" + "-"*50)
        
        # Chuyển đổi tài liệu thành văn bản
        text = extract_text_from_file(file_path)

        # Trích xuất thông tin y khoa
        extracted_info = extract_medical_info(text)
        
        print("\n🔍 Thông tin trích xuất:")
        for key, value in extracted_info.items():
            print(f"{key}: {value if value else 'Không có dữ liệu'}")

        # Kiểm tra dữ liệu bị thiếu
        missing_sections = check_missing_data(extracted_info)

        if missing_sections:
            print(f"\n Dữ liệu bị thiếu ở các mục: {', '.join(missing_sections)}")
        else:
            print("\n✅ Tài liệu đầy đủ.")

        print("="*80)

# Chạy xử lý trên thư mục chứa file tài liệu
directory_path = "documents"
process_multiple_files(directory_path)
