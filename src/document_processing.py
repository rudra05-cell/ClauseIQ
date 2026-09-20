"""Extracts text from uploaded contracts and splits it into clause-level sections."""

import re
import io

CLAUSE_HEADER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d{1,2}(?:\.\d{1,2})*\.?|\([a-z]\)|Section\s+\d+[:.]?)\s+",
    re.IGNORECASE,
)


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(uploaded_file)
    if name.endswith(".docx"):
        return _extract_docx(uploaded_file)
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {name}")


def _extract_pdf(uploaded_file) -> str:
    import pdfplumber

    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n\n".join(p for p in pages if p)


def _extract_docx(uploaded_file) -> str:
    import docx

    doc = docx.Document(io.BytesIO(uploaded_file.read()))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def split_into_clauses(text: str, min_length: int = 40, max_length: int = 1500) -> list:
    """Splits contract text into clause-sized chunks using numbered headers where
    present, falling back to paragraph breaks and sentence-level splitting for
    any section that remains too long."""
    text = text.replace("\r\n", "\n")

    chunks = [c.strip() for c in CLAUSE_HEADER_PATTERN.split(text) if c and c.strip()]
    if len(chunks) <= 1:
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    clauses = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            if len(chunk) >= min_length:
                clauses.append(chunk)
            continue

        sentences = re.split(r"(?<=[.;])\s+(?=[A-Z])", chunk)
        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) < max_length:
                buffer += " " + sentence
            else:
                if len(buffer.strip()) >= min_length:
                    clauses.append(buffer.strip())
                buffer = sentence
        if len(buffer.strip()) >= min_length:
            clauses.append(buffer.strip())

    return clauses
