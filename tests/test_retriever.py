import pytest

from utils.config import load_settings
from utils.retriever import Retriever


@pytest.mark.integration
def test_retriever_returns_real_matches() -> None:
    settings = load_settings()

    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not configured.")
    if not settings.chroma_dir.exists():
        pytest.skip("Chroma directory does not exist yet.")

    retriever = Retriever(settings)
    if retriever.collection.count() == 0:
        pytest.skip("Chroma collection is empty. Ingest data first.")

    results = retriever.search("What does Paul Graham say about great work?", top_k=3)

    print("\nRetrieved matches:")
    for match in results:
        print(f"- chunk_id: {match['chunk_id']}")
        print(f"  title: {match['title']}")
        print(f"  doc_id: {match['doc_id']}")
        print(f"  chunk_index: {match['chunk_index']}")
        print(f"  url: {match['url']}")
        print(f"  distance: {match['distance']}")
        print(f"  text: {str(match['text'])[:200]}")
        print()

    assert results
    assert len(results) <= 3

    first = results[0]
    assert first["chunk_id"]
    assert first["text"]
    assert first["doc_id"]
    assert first["title"]
    assert first["url"]
    assert isinstance(first["chunk_index"], int)
