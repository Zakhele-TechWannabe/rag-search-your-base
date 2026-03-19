from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
import time
import uvicorn

from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi import HTTPException
from fastapi import Query

from utils.config import load_settings
from utils.logger import get_logger
from core.chunking import CHUNKS_PATH, chunk_documents, write_chunks
from core.embed import upsert_chunks
from core.evaluate import evaluate_request_record
from core.generate import generate_answer
from core.ingest import (
    CATALOG_PATH,
    ESSAY_DIR,
    discover_catalog,
    ingest_documents,
    read_list,
)
from utils.clients import ChromaClient


startup_logger = get_logger("startup", folder="system")
discover_logger = get_logger("discover", folder="endpoints")
ingest_logger = get_logger("ingest", folder="endpoints")
ask_logger = get_logger("ask", folder="endpoints")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    startup_logger.info("Application startup")
    if not CATALOG_PATH.exists():
        startup_logger.info("Catalog missing, running initial discovery")
        discover_catalog()
    chroma_client = ChromaClient(path=str(settings.chroma_dir))
    chroma_client.heartbeat()
    chroma_client.get_or_create_collection(settings.chroma_collection_name)
    startup_logger.info(
        "Chroma ready at %s with collection %s",
        settings.chroma_dir,
        settings.chroma_collection_name,
    )
    yield


app = FastAPI(
    title="RAG Search Your Base",
    description="A Retrieval-Augmented Generation (RAG) system that can answer questions over a provided knowledge base.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "rag-search-your-base",
        "endpoints": ["/health", "/discover", "/ingest", "/ask"],
    }


@app.get("/health")
def status() -> dict[str, object]:
    settings = app.state.settings
    startup_logger.info("Health endpoint called")
    return {
        "catalog": str(CATALOG_PATH),
        "essays": str(ESSAY_DIR),
        "chunks": str(CHUNKS_PATH),
        "chroma_dir": str(settings.chroma_dir),
        "results": str(settings.results_dir),
        "status": {
            "catalog_file": "present" if CATALOG_PATH.exists() else "missing",
            "essay_dir": "present" if ESSAY_DIR.exists() else "missing",
            "chunks_file": "present" if CHUNKS_PATH.exists() else "missing",
            "chroma_dir": "present" if settings.chroma_dir.exists() else "missing",
        },
    }


@app.post("/discover")
def discover() -> dict[str, str]:
    discover_logger.info("Discover endpoint called")
    catalog_path = discover_catalog()
    discover_logger.info("Catalog saved to %s", catalog_path)
    return {"message": "Catalog saved.", "catalog_path": str(catalog_path)}


@app.post("/ingest")
def ingest(
    slug: str | None = None,
    position: int | None = Query(default=None, ge=0),
    start: int | None = Query(default=None, ge=0),
    end: int | None = Query(default=None),
) -> dict[str, object]:
    settings = app.state.settings
    if not settings.openai_api_key:
        ingest_logger.warning("Ingest called without OPENAI_API_KEY")
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")

    essays = read_list()
    max_index = len(essays)
    if end is not None and end > max_index:
        ingest_logger.warning("End %s exceeds catalog size %s", end, max_index)
        raise HTTPException(
            status_code=400,
            detail=f"`end` cannot be greater than {max_index}.",
        )
    if start is not None and end is not None and end < start:
        ingest_logger.warning("Invalid ingest range start=%s end=%s", start, end)
        raise HTTPException(
            status_code=400,
            detail="`end` cannot be lower than `start`.",
        )

    ingest_logger.info(
        "Ingest endpoint called with slug=%s position=%s start=%s end=%s",
        slug,
        position,
        start,
        end,
    )
    documents = ingest_documents(slug=slug, position=position, start=start, end=end)
    chunks = chunk_documents(docs=documents)
    write_chunks(chunks, path=settings.chunks_path)
    upserted = upsert_chunks(chunks, settings)
    ingest_logger.info(
        "Ingested %s documents, wrote %s chunks, upserted %s chunks",
        len(documents),
        len(chunks),
        upserted,
    )
    return {
        "message": "Ingestion, chunking, and Chroma upsert complete.",
        "filters": {
            "slug": slug,
            "position": position,
            "start": start,
            "end": end,
        },
        "documents_ingested": len(documents),
        "chunks_written": len(chunks),
        "chunks_upserted": upserted,
        "chunks_path": str(settings.chunks_path),
        "collection": settings.chroma_collection_name,
    }


@app.post("/ask")
def ask(
    background_tasks: BackgroundTasks,
    question: str = Query(..., min_length=1),
) -> dict[str, object]:
    settings = app.state.settings
    if not settings.openai_api_key:
        ask_logger.warning("Ask called without OPENAI_API_KEY")
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")

    ask_logger.info("Ask endpoint called with question=%s", question)
    started_at = time.perf_counter()
    result = generate_answer(question, settings)
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    ask_day_dir = settings.results_dir / "ask" / day
    ask_requests_dir = ask_day_dir / "requests"
    ask_requests_dir.mkdir(parents=True, exist_ok=True)
    result_path = ask_requests_dir / f"{timestamp}.json"

    request_record = {
        "timestamp": timestamp,
        "question": question,
        "duration_ms": duration_ms,
        "output": {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "confidence_score": result.get("confidence_score", 0),
        },
        "internal": result,
    }
    evaluation_result = {
        "retrieval": {
            "expected_slug": "unavailable",
            "retrieved_slugs": [],
            "hit": None,
        },
        "answer_evaluation": {
            "expected_answer_summary": "unavailable",
            "alignment_score": None,
            "alignment_reason": "Background evaluation has not completed yet.",
        },
        "judge": {
            "retrieval_assessment": "Background evaluation has not completed yet.",
            "answer_assessment": "Background evaluation has not completed yet.",
            "overall_note": "Background evaluation has not completed yet.",
        },
    }
    record_payload = {
        "request_record": request_record,
        "evaluation_result": evaluation_result,
    }
    result_path.write_text(json.dumps(record_payload, indent=2) + "\n", encoding="utf-8")
    background_tasks.add_task(evaluate_request_record, result_path)

    summary_path = ask_day_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "date": day,
            "total_requests": 0,
            "average_confidence_score": 0,
            "average_duration_ms": 0,
            "retry_count": 0,
            "requests": [],
        }

    summary["requests"].append(
        {
            "timestamp": timestamp,
            "question": question,
            "confidence_score": result.get("confidence_score", 0),
            "duration_ms": duration_ms,
            "needs_retry": result.get("needs_retry", False),
            "citation_count": len(result.get("citations", [])),
            "evaluation_summary": {
                "recall_at_k": None,
                "alignment_score": None,
                "overall_note": "Background evaluation has not completed yet.",
            },
        }
    )
    summary["total_requests"] = len(summary["requests"])
    summary["average_confidence_score"] = round(
        sum(item["confidence_score"] for item in summary["requests"]) / summary["total_requests"],
        2,
    )
    summary["average_duration_ms"] = round(
        sum(item["duration_ms"] for item in summary["requests"]) / summary["total_requests"],
        2,
    )
    summary["retry_count"] = sum(1 for item in summary["requests"] if item["needs_retry"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ask_logger.info(
        "Ask completed with confidence_score=%s needs_retry=%s duration_ms=%s",
        result.get("confidence_score", 0),
        result.get("needs_retry", True),
        duration_ms,
    )
    ask_logger.info("Full ask result saved to %s", result_path)
    ask_logger.info("Queued background evaluation for %s", result_path)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "confidence_score": result.get("confidence_score", 0),
    }


def main() -> None:
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )


if __name__ == "__main__":
    main()
