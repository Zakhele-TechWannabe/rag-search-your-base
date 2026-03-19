from utils.clients import ChromaClient, LLMClient
from utils.config import Settings
from utils.logger import get_logger


retriever_logger = get_logger("retriever", folder="pipeline")


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm_client = LLMClient(api_key=settings.openai_api_key)
        self.chroma_client = ChromaClient(path=str(settings.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            settings.chroma_collection_name
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | int | float]]:
        retriever_logger.info("Running retrieval for query=%s top_k=%s", query, top_k)
        query_embedding = self.llm_client.create_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index],
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "distance": distances[index] if index < len(distances) else 0.0,
                }
            )

        retriever_logger.info("Retrieved %s matches", len(matches))
        retriever_logger.debug("Retrieved matches: %s", matches)
        return matches
