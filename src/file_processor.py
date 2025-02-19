import os
import docx
import PyPDF2
from email import policy
from email.parser import BytesParser
from .ocr import ocr_from_image
from .utils import extract_archive


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)


def extract_text_from_pdf(file_path, ocr_enabled=True):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    if ocr_enabled and len(text.strip()) < 10:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(file_path)
            for image in images:
                text += ocr_from_image(image)
        except Exception as e:
            print(f"OCR on PDF failed for {file_path}: {e}")
    return text


def extract_text_from_eml(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_content()
    else:
        text = msg.get_content()
    return text


def extract_text_from_file(file_path, ocr_enabled=True):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".docx"]:
        return extract_text_from_docx(file_path)
    elif ext in [".pdf"]:
        return extract_text_from_pdf(file_path, ocr_enabled=ocr_enabled)
    elif ext in [".eml"]:
        return extract_text_from_eml(file_path)
    else:
        try:
            return extract_text_from_txt(file_path)
        except Exception as e:
            print(f"Cannot process file {file_path}: {e}")
            return ""
