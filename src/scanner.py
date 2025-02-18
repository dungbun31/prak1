import os
import argparse
import json
from tqdm import tqdm
from .file_processor import extract_text_from_file
from .utils import is_archive, extract_archive
from .classifier import DocumentClassifier


def scan_directory(directory, classifier, ocr_enabled=True):
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Nếu file là archive, giải nén và xử lý nội dung bên trong
            if is_archive(file_path):
                extracted_path = extract_archive(file_path)
                if extracted_path:
                    extracted_results = scan_directory(
                        extracted_path, classifier, ocr_enabled
                    )
                    results.extend(extracted_results)
            else:
                try:
                    text = extract_text_from_file(file_path, ocr_enabled=ocr_enabled)
                    if text.strip():
                        label, score = classifier.classify(text)
                        results.append(
                            {"file_path": file_path, "label": label, "score": score}
                        )
                    else:
                        results.append(
                            {
                                "file_path": file_path,
                                "label": "No text extracted",
                                "score": 0.0,
                            }
                        )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    results.append(
                        {"file_path": file_path, "label": "Error", "score": 0.0}
                    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scan Doc Dung LLM - Document Scanner and Classifier"
    )
    # Nếu không cung cấp đối số, mặc định quét thư mục "data"
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="data",
        help="Đường dẫn tới thư mục hoặc file cần quét (mặc định: data)",
    )
    # Nếu không chỉ định model qua đối số, đọc từ config.json
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Đường dẫn tới model phân loại hoặc model identifier từ Hugging Face. Nếu không chỉ định, sẽ dùng model từ config.json",
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Đường dẫn tới file config"
    )
    parser.add_argument(
        "--ocr", action="store_true", help="Kích hoạt OCR cho file không chứa văn bản"
    )
    parser.add_argument(
        "--output", type=str, default="results.json", help="Đường dẫn file kết quả"
    )
    args = parser.parse_args()

    # Nếu không truyền model qua đối số, đọc model_path từ config.json
    model_path = args.model
    if not model_path:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
                model_path = config.get(
                    "model_path", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
                )
        except Exception as e:
            print(f"Error reading config: {e}")
            model_path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    # Khởi tạo classifier
    classifier = DocumentClassifier(model_path=model_path, config_path=args.config)

    # Xử lý nếu đường dẫn là thư mục hay file
    if os.path.isdir(args.path):
        results = scan_directory(args.path, classifier, ocr_enabled=args.ocr)
    else:
        text = extract_text_from_file(args.path, ocr_enabled=args.ocr)
        label, score = classifier.classify(text)
        results = [{"file_path": args.path, "label": label, "score": score}]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Quét hoàn tất. Kết quả lưu tại {args.output}")


if __name__ == "__main__":
    main()
