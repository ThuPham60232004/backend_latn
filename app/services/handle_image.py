import os
import glob
import fitz  
import docx  
import re  
import ftfy  
import unicodedata  
import google.generativeai as genai
from unidecode import unidecode  
import json
import requests
import csv
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENALEX_API_URL = "https://api.openalex.org/works" 

if not GEMINI_API_KEY:
    raise ValueError("API key is missing. Set GEMINI_API_KEY in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

def scan_directory(directory_path):
    return glob.glob(os.path.join(directory_path, "*.pdf")) + \
           glob.glob(os.path.join(directory_path, "*.docx")) + \
           glob.glob(os.path.join(directory_path, "*.txt"))

def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        return clean_text(text)
    except Exception as e:
        return f"Lỗi đọc PDF {file_path}: {e}"

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return clean_text(text)
    except Exception as e:
        return f"Lỗi đọc DOCX {file_path}: {e}"

def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return clean_text(file.read())
    except Exception as e:
        return f"Lỗi đọc TXT {file_path}: {e}"

def clean_text(text):
    """Chuẩn hóa văn bản"""
    text = ftfy.fix_text(text)  
    text = unicodedata.normalize("NFC", text)  
    text = unidecode(text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = re.sub(r'[^\w\s,.\-]', '', text)  
    return text

def fetch_medical_info_from_openalex(query):
    """Tìm kiếm thông tin y khoa từ OpenAlex API"""
    params = {"search": query, "filter": "type:journal-article", "per_page": 1}
    try:
        response = requests.get(OPENALEX_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and data["results"]:
            result = data["results"][0]
            return {
                "title": result.get("title", "Không có tiêu đề"),
                "abstract": clean_text(result.get("abstract", "Không có mô tả")),
                "doi": result.get("doi", "Không có DOI"),
                "url": result.get("id", "Không có URL")
            }
        return None
    except Exception as e:
        return f"Lỗi OpenAlex: {e}"

def extract_medical_info(text):
    """Trích xuất thông tin y khoa từ văn bản"""
    prompt = f"""
    Hãy trích xuất thông tin y khoa từ văn bản dưới dạng JSON hợp lệ **không chứa Markdown**.
    Định dạng đầu ra **chỉ bao gồm JSON**:

    {{
        "Tên bệnh": "",
        "Triệu chứng": "",
        "Vị trí xuất hiện": "",
        "Nguyên nhân": "",
        "Tiêu chí chẩn đoán": "",
        "Chẩn đoán phân biệt": "",
        "Điều trị": "",
        "Phòng bệnh": ""
    }}

    - Nếu không có thông tin, đặt giá trị `"Không tìm thấy"`.  
    - **Không thêm giải thích, không in Markdown, không thêm ký tự thừa**.  

    Văn bản cần trích xuất:  
    {text}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-002")
        response = model.generate_content(prompt)

        raw_text = response.text.strip()
        raw_text = re.sub(r"^```json\n|\n```$", "", raw_text)
        extracted_info = json.loads(raw_text)

        for key, value in extracted_info.items():
            if value == "Không tìm thấy" or len(value) < 10:
                medical_info = fetch_medical_info_from_openalex(extracted_info["Tên bệnh"])
                if medical_info:
                    extracted_info[key] = medical_info["abstract"]
        
        return extracted_info

    except json.JSONDecodeError:
        print("❌ Lỗi: Không thể parse JSON từ Gemini.")
        return {}
    except Exception as e:
        print(f"❌ Lỗi trích xuất thông tin y khoa: {e}")
        return {}

def handlefile():
    directory_path = "app/static/handlefile"
    if not os.path.exists(directory_path):
        print("Thư mục không tồn tại. Vui lòng kiểm tra lại.")
        return
    
    files = scan_directory(directory_path)
    if not files:
        print("Không tìm thấy tài liệu nào.")
        return

    for file in files:
        print(f"\nĐọc nội dung: {file}")
        if file.endswith(".pdf"):
            content = read_pdf(file)
        elif file.endswith(".docx"):
            content = read_docx(file)
        elif file.endswith(".txt"):
            content = read_txt(file)
        else:
            content = "Định dạng không hỗ trợ."

        print("\nTrích xuất thông tin y khoa:")
        extracted_info = extract_medical_info(content)
        print(json.dumps(extracted_info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    handlefile()
