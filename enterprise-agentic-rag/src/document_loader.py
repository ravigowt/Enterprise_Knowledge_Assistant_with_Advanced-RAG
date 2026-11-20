import os
from typing import List
from langchain_core.documents import Document

import pypdf
import docx

def text_to_doc(text: str, source: str = "inline") -> Document:
    return Document(page_content = text, metadata = {"source": source})

def load_txt(path: str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding = "utf-8", errors = "ignore") as f:
        text = f.read()
    return [text_to_doc(text, source = os.path.abspath(path))]

def load_pdf(path: str) -> List[Document]:
    reader = pypdf.PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(text_to_doc(text, source = f"{os.path.abspath(path)}: page: {i+1}"))
    return docs

def load_docx(path: str) -> List[Document]:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    return [text_to_doc(text, source = os.path.abspath(path))]

def load_document(path_or_url: str) -> List[Document]:
    ext = os.path.splitext(path_or_url)[1].lower()

    if ext in [".txt", "md"]:
        return load_txt(path_or_url)
    elif ext == ".pdf":
        return load_pdf(path_or_url)
    elif ext in [".docx", ".docs", ".doc"]:
        return load_docx(path_or_url)
    else:
        return "No Supporting Documents"
