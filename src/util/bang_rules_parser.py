import os
import re
from typing import Dict, List


LATEX_FILE_EXTENSIONS = {".tex"}


def parse_bang_rules(document_paths: List[str]) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for document_path in sorted(document_paths):
        if os.path.splitext(document_path)[1].lower() not in LATEX_FILE_EXTENSIONS:
            continue

        with open(document_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        chunks.extend(_extract_general_rules(raw_text, document_path))
        chunks.extend(_extract_card_captions(raw_text, document_path))

    return chunks


def _extract_general_rules(raw_text: str, source_path: str) -> List[Dict[str, str]]:
    text = _strip_comments(raw_text)
    chapter = _extract_heading(text, "chapter")
    sections = re.findall(r"\\section\*?\{([^}]*)\}", text)
    chunks: List[Dict[str, str]] = []

    enumerate_blocks = re.findall(
        r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", text, flags=re.DOTALL
    )
    for block_index, block in enumerate(enumerate_blocks, start=1):
        rules = re.findall(r"\\item\s+(.*?)(?=(\\item|$))", block, flags=re.DOTALL)
        normalized_rules = [
            _normalize_text(rule_text)
            for rule_text, _ in rules
            if _normalize_text(rule_text)
        ]
        if not normalized_rules:
            continue

        heading_parts = [part for part in [chapter, *sections] if part]
        title = " / ".join(heading_parts) if heading_parts else os.path.basename(source_path)
        content = "\n".join(
            [
                f"Zdroj pravidiel: {title}",
                "Vseobecne pravidla:",
                *[f"- {rule}" for rule in normalized_rules],
            ]
        )
        chunks.append(
            {
                "content": content,
                "source": f"{os.path.relpath(source_path)}:rules:{block_index}",
            }
        )

    glossary_matches = re.findall(
        r"\\textbf\{([^}]*)\}\s*&\s*(.*?)(?=\\\\)",
        text,
        flags=re.DOTALL,
    )
    for term, explanation in glossary_matches:
        normalized_term = _normalize_text(term)
        normalized_explanation = _normalize_text(explanation)
        if not normalized_term or not normalized_explanation:
            continue

        content = "\n".join(
            [
                "Slovnik pravidiel Bang!",
                f"Pojem: {normalized_term}",
                f"Vysvetlenie: {normalized_explanation}",
            ]
        )
        chunks.append(
            {
                "content": content,
                "source": f"{os.path.relpath(source_path)}:glossary:{normalized_term}",
            }
        )

    return chunks


def _extract_card_captions(raw_text: str, source_path: str) -> List[Dict[str, str]]:
    text = _strip_comments(raw_text)
    chapter = _extract_heading(text, "chapter")
    section = _extract_heading(text, "section")
    chunks: List[Dict[str, str]] = []

    for index, (short_title, body) in enumerate(_iter_captions(text), start=1):
        title = _normalize_text(short_title)
        description = _normalize_text(body)
        if not title or not description:
            continue

        heading_parts = [part for part in [chapter, section] if part]
        heading = " / ".join(heading_parts) if heading_parts else "Bang! pravidla"
        content = "\n".join(
            [
                f"Zdroj pravidiel: {heading}",
                f"Karta alebo pojem: {title}",
                f"Popis pravidla: {description}",
            ]
        )
        chunks.append(
            {
                "content": content,
                "source": f"{os.path.relpath(source_path)}:caption:{index}:{title}",
            }
        )

    return chunks


def _iter_captions(text: str) -> List[tuple[str, str]]:
    captions: List[tuple[str, str]] = []
    cursor = 0

    while True:
        caption_start = text.find(r"\caption", cursor)
        if caption_start == -1:
            break

        cursor = caption_start + len(r"\caption")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        short_title = ""
        if cursor < len(text) and text[cursor] == "[":
            short_title, cursor = _read_balanced_block(text, cursor, "[", "]")
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1

        if cursor >= len(text) or text[cursor] != "{":
            continue

        body, cursor = _read_balanced_block(text, cursor, "{", "}")
        captions.append((short_title, body))

    return captions


def _read_balanced_block(
    text: str, start_index: int, open_char: str, close_char: str
) -> tuple[str, int]:
    depth = 0
    content_start = start_index + 1

    for index in range(start_index, len(text)):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1

    return text[content_start:], len(text)


def _extract_heading(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\*?\{{([^}}]*)\}}", text)
    if not match:
        return ""
    return _normalize_text(match.group(1))


def _strip_comments(text: str) -> str:
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)%.*", "", text)
    return text


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\\\\", "\n")
    cleaned = re.sub(r"\\cite\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\(chapter|section)\*?\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\(begin|end)\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\label\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\url\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\textit\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\emph\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", cleaned)
    cleaned = cleaned.replace("~", " ")
    cleaned = cleaned.replace("$", " ")
    cleaned = cleaned.replace("&", " ")
    cleaned = cleaned.replace("{", " ")
    cleaned = cleaned.replace("}", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
