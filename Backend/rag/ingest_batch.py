# Backend/rag/ingest_batch.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ..config import settings

DOCS_DIR = Path(settings.DOCS_DIR)
INDEX_DIR = Path(settings.RAG_INDEX_DIR)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()  # page별 Document, metadata에 page 포함
    # source 경로 메타 보강
    for d in docs:
        d.metadata["source"] = str(path)
    return docs


def load_txt(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = str(path)
    return docs


def load_xlsx(path: Path) -> List[Document]:
    docs: List[Document] = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        text = df.to_csv(index=False)
        meta = {"source": str(path), "sheet": sheet}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def gather_documents(root: Path) -> List[Document]:
    out: List[Document] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            out.extend(load_pdf(p))
        elif ext in (".txt",):
            out.extend(load_txt(p))
        elif ext in (".xlsx", ".xls", ".xlsm", ".csv"):
            # csv도 pandas로 처리
            if ext == ".csv":
                df = pd.read_csv(p)
                text = df.to_csv(index=False)
                out.append(Document(page_content=text, metadata={"source": str(p), "sheet": "csv"}))
            else:
                out.extend(load_xlsx(p))
        else:
            # 무시
            pass
    return out


def main():
    print(f"[ingest] docs dir: {DOCS_DIR}")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw_docs = gather_documents(DOCS_DIR)
    print(f"[ingest] loaded docs: {len(raw_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    print(f"[ingest] chunks: {len(chunks)}")

    emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    if (INDEX_DIR / "index.faiss").exists():
        vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
        vs.save_local(str(INDEX_DIR))
        print("[ingest] appended to existing index.")
    else:
        vs = FAISS.from_documents(chunks, emb)
        vs.save_local(str(INDEX_DIR))
        print("[ingest] created new index.")


if __name__ == "__main__":
    main()