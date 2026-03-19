import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger

from .ingest import EssayDocument


CHUNKS_PATH = Path("data/chunks/recursive_chunks.json")
chunk_logger = get_logger("chunking", folder="pipeline")


def chunk_document(
    doc: EssayDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, str | int]]:
    chunk_logger.info(
        "Chunking document %s with size=%s overlap=%s",
        doc.slug,
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_text(doc.text)

    chunks = []
    for index, text in enumerate(texts, start=1):
        chunks.append(
            {
                "chunk_id": f"{doc.slug}_{index:03d}",
                "chunk_index": index,
                "doc_id": doc.slug,
                "title": doc.title,
                "url": doc.url,
                "text": text,
            }
        )

    chunk_logger.info("Created %s chunks for %s", len(chunks), doc.slug)
    return chunks


def chunk_documents(
    docs: list[EssayDocument] | None = None,
    essay_dir: Path = Path("data/essays"),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, str | int]]:
    all_chunks = []

    if docs is None:
        docs = []
        for path in sorted(essay_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            docs.append(EssayDocument(**payload))

    for doc in docs:
        chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    chunk_logger.info("Prepared %s chunks in total", len(all_chunks))
    return all_chunks


def write_chunks(chunks: list[dict[str, str | int]], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")
    chunk_logger.info("Wrote %s chunks to %s", len(chunks), path)
