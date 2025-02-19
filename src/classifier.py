import os
import json
from transformers import AutoTokenizer, pipeline


class DocumentClassifier:
    def __init__(self, model_path, config_path=None, token=None):
        self.candidate_labels = []
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                categories = config.get("categories", {})
                self.candidate_labels = list(categories.values())
        if not self.candidate_labels:
            print(
                "Warning: Candidate labels not provided in config. Classification may not work as expected."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, use_auth_token=token
        )
        self.nlp = pipeline(
            "zero-shot-classification",
            model=model_path,
            tokenizer=tokenizer,
            use_auth_token=token,
        )

    def classify(self, text):
        if not self.candidate_labels:
            return "Unknown", 0.0
        result = self.nlp(text, candidate_labels=self.candidate_labels)
        if result and "labels" in result and "scores" in result:
            best_idx = result["scores"].index(max(result["scores"]))
            label = result["labels"][best_idx]
            score = result["scores"][best_idx]
            return label, score
        return None, 0.0
