import os
import zipfile
import rarfile
import py7zr
import tempfile


def is_archive(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in [".zip", ".rar", ".7z"]


def extract_archive(file_path, extract_to=None):
    if extract_to is None:
        extract_to = tempfile.mkdtemp()
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".zip":
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif ext == ".rar":
            with rarfile.RarFile(file_path, "r") as rar_ref:
                rar_ref.extractall(extract_to)
        elif ext == ".7z":
            with py7zr.SevenZipFile(file_path, mode="r") as seven_z:
                seven_z.extractall(path=extract_to)
        return extract_to
    except Exception as e:
        print(f"Error extracting archive {file_path}: {e}")
        return None
