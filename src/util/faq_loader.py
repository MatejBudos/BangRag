import json
import os
import re
from typing import Dict, List

import fitz


def load_evaluation_questions(eval_path: str) -> List[Dict[str, str]]:
    if os.path.isfile(eval_path) and eval_path.lower().endswith(".json"):
        with open(eval_path, "r", encoding="utf-8") as file:
            return json.load(file)

    pdf_paths: List[str] = []
    if os.path.isfile(eval_path) and eval_path.lower().endswith(".pdf"):
        pdf_paths = [eval_path]
    elif os.path.isdir(eval_path):
        for root, _, files in os.walk(eval_path):
            for file_name in files:
                if file_name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, file_name))

    if not pdf_paths:
        raise ValueError(f"Unsupported evaluation input: {eval_path}")

    questions: List[Dict[str, str]] = []
    for pdf_path in sorted(pdf_paths):
        questions.extend(_parse_faq_pdf(pdf_path))

    return questions


def _parse_faq_pdf(pdf_path: str) -> List[Dict[str, str]]:
    document = fitz.open(pdf_path)
    try:
        raw_text = "\n".join(page.get_text() for page in document)
    finally:
        document.close()

    normalized = raw_text.replace("\xa0", " ")
    normalized = re.sub(r"\r", "\n", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)

    qa_blocks = re.findall(r"(Q\d+\..*?)(?=Q\d+\.|$)", normalized, flags=re.DOTALL)
    results: List[Dict[str, str]] = []
    for block in qa_blocks:
        match = re.match(r"Q\d+\.\s*(.*?)\s*A\.\s*(.*)", block, flags=re.DOTALL)
        if not match:
            continue

        question = _normalize_qa_text(match.group(1))
        answer = _normalize_qa_text(match.group(2))
        if not question or not answer:
            continue

        results.append({"question": question, "answer": answer})

    return results


def _normalize_qa_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"(F\.A\.Q\.|Frequently Asked Questions|BANG! DODGE CITY: F\.A\.Q\.|© MMXVI.*|www\.dvgiochi\.com)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()
