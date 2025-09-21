"""Ingest Bangla agricultural PDFs/DOCX files into a simple retrieval index."""

import argparse
import os
from pathlib import Path
from typing import List

import fitz  # pymupdf
from dotenv import load_dotenv

import google.generativeai as genai
from docx import Document

from rag_utils import (
    embed_texts,
    ensure_dir,
    normalize_bangla_text,
    paragraph_chunks,
    save_index,
)


def extract_pdf(path: Path) -> List[dict]:
    records: List[dict] = []
    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = normalize_bangla_text(text)
            if not text:
                continue
            for chunk in paragraph_chunks(text):
                records.append(
                    {
                        "source": path.name,
                        "page": page_idx,
                        "text": chunk,
                    }
                )
    return records


def extract_docx(path: Path) -> List[dict]:
    records: List[dict] = []
    doc = Document(path)
    paragraphs = []
    for para in doc.paragraphs:
        norm = normalize_bangla_text(para.text)
        if norm:
            paragraphs.append(norm)
    text = "\n".join(paragraphs)
    if not text:
        return records
    for idx, chunk in enumerate(paragraph_chunks(text), start=1):
        records.append(
            {
                "source": path.name,
                "page": idx,
                "text": chunk,
            }
        )
    return records


def ingest(data_dir: Path, out_dir: Path) -> None:
    input_paths = sorted(
        [p for p in data_dir.glob("**/*.pdf") if p.is_file()] +
        [p for p in data_dir.glob("**/*.docx") if p.is_file()]
    )
    if not input_paths:
        raise SystemExit(f"No PDF or DOCX files found under {data_dir}")

    all_records: List[dict] = []
    for path in input_paths:
        print(f"Extracting {path}…")
        if path.suffix.lower() == ".pdf":
            all_records.extend(extract_pdf(path))
        else:
            all_records.extend(extract_docx(path))

    if not all_records:
        raise SystemExit("No text extracted from documents.")

    print(f"Embedding {len(all_records)} chunks…")
    texts = [rec["text"] for rec in all_records]
    embeddings = embed_texts(texts)

    ensure_dir(out_dir)
    save_index(embeddings, all_records, out_dir=out_dir)
    print(f"Saved index to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Bangla RAG index from PDFs")
    parser.add_argument("--data-dir", default="data/pdfs", help="Directory containing PDF/DOCX files")
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory to write embeddings/metadata (defaults to data/)",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY must be set in environment or .env")
    genai.configure(api_key=api_key)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    ingest(data_dir, out_dir)


if __name__ == "__main__":
    main()
