import pytesseract
from PIL import Image


def ocr_from_image(image, lang="eng+rus+vie"):
    """
    Nếu image là đường dẫn thì mở ảnh, nếu là PIL Image thì sử dụng trực tiếp.
    Sử dụng pytesseract để trích xuất văn bản với các ngôn ngữ:
      - eng: tiếng Anh
      - rus: tiếng Nga
      - vie: tiếng Việt
    """
    if isinstance(image, str):
        image = Image.open(image)
    text = pytesseract.image_to_string(image, lang=lang)
    return text
