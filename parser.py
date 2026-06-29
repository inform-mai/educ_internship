import os
import pdfplumber

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f: text = f.read()
    return text

def read_pdf(file_path):
    full_text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text is not None:
                full_text += page_text + "\n"

    return full_text

def parse_file(file_path):
    # кидаем файл, получаем текст
    
    if not os.path.exists(file_path):
        print(f"файл {file_path} не найден")
        return None
    if file_path.endswith(".txt"):
        text = read_txt(file_path)
    elif file_path.endswitch(".pdf"):
        text = read_pdf(file_path)
    else:
        print("нужен .txt или .pdf")
        return None
    
    print(f"{file_path} прочитан, {len(text)} символов")
    return text