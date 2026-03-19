from utils.clients import ChromaClient, LLMClient
from utils.config import Settings
from utils.logger import get_logger


embed_logger = get_logger("embed", folder="pipeline")


def upsert_chunks(
    chunks: list[dict[str, str | int]],
    settings: Settings,
) -> int:
    if not chunks:
        embed_logger.warning("No chunks provided for upsert")
        return 0

    embed_logger.info("Upserting %s chunks into Chroma", len(chunks))
    llm_client = LLMClient(api_key=settings.openai_api_key)
    chroma_client = ChromaClient(path=str(settings.chroma_dir))
    collection = chroma_client.get_or_create_collection(
        settings.chroma_collection_name
    )

    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    documents = [str(chunk["text"]) for chunk in chunks]
    metadatas = [
        {
            "doc_id": str(chunk["doc_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "title": str(chunk["title"]),
            "url": str(chunk["url"]),
        }
        for chunk in chunks
    ]
    embeddings = [llm_client.create_embedding(text) for text in documents]
    embed_logger.info("Generated %s embeddings", len(embeddings))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    embed_logger.info(
        "Upserted %s chunks into collection %s",
        len(ids),
        settings.chroma_collection_name,
    )
    return len(ids)
