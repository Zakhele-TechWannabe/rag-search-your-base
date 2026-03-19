# Architecture Notes

This document describes the current MVP architecture of the project as implemented, not the original plan.

## Overview

The system is a small RAG pipeline over the Paul Graham essays. It has four main flows:

1. startup and catalog discovery
2. ingestion, chunking, and vector upsert
3. question answering
4. evaluation

The application is exposed through FastAPI in [src/main.py](../src/main.py).

## High-Level Flow

### 1. Startup

On startup the app:

* loads settings from environment variables
* ensures the essay catalog exists, and runs discovery if it does not
* initializes a persistent local Chroma client
* ensures the configured Chroma collection exists

This keeps the first-run experience simple: the catalog is prepared automatically, and Chroma is ready before requests start coming in.

### 2. Ingest Flow

The ingest flow is triggered through `POST /ingest`.

It performs these steps in sequence:

1. read the discovered catalog
2. optionally filter essays by `slug`, `position`, or `start` / `end`
3. fetch the selected essays
4. parse the essay body into normalized local JSON documents
5. chunk those documents using recursive character chunking
6. write the combined chunk file locally
7. embed the chunks with OpenAI embeddings
8. upsert the chunk embeddings and metadata into Chroma

This keeps ingestion and indexing as one simple pipeline step instead of making the user run several separate commands.

### 3. Ask Flow

The answer flow is triggered through `POST /ask`.

It performs these steps:

1. retrieve candidate chunks from Chroma using dense vector search
2. rerank the retrieved chunks with an LLM
3. select a bounded subset of reranked chunks as answer context
4. run a reflection step to decide whether the context is sufficient
5. generate a final grounded answer using only the selected context
6. return a slim public response:
   * `answer`
   * `citations`
   * `confidence_score`
7. save a fuller internal request artifact to `results/ask/<day>/requests/<timestamp>.json`
8. queue background evaluation for that request

The idea is to keep the API response simple while still preserving the internal trace needed for debugging and evaluation.

### 4. Evaluation Flow

There are two evaluation paths:

* manual evaluation over the full eval set in [docs/eval_set.json](eval_set.json)
* background evaluation for individual `/ask` requests using [docs/ask_eval_set.json](ask_eval_set.json)

The evaluator combines:

* a hard retrieval metric: `Recall@3`
* a Gemini judge for qualitative interpretation

The judge is not used as the primary metric. It only adds commentary around retrieval relevance and answer alignment.

## Main Components

### API Layer

The FastAPI application lives in [src/main.py](../src/main.py).

Key endpoints:

* `POST /discover`
* `POST /ingest`
* `POST /ask`
* `GET /health`

### Ingestion

The ingestion logic lives in [src/core/ingest.py](../src/core/ingest.py).

Main responsibilities:

* fetch the Paul Graham index page
* parse essay links
* fetch essay pages
* parse essay title and body text
* write the catalog and essay JSON artifacts

The ingestion layer is intentionally simple and uses HTML parsing rather than a more complex crawler or scraper framework.

### Chunking

The chunking logic lives in [src/core/chunking.py](../src/core/chunking.py).

It uses LangChain's `RecursiveCharacterTextSplitter` with:

* chunk size: `1000`
* chunk overlap: `200`

Each chunk includes metadata such as:

* `chunk_id`
* `chunk_index`
* `doc_id`
* `title`
* `url`
* `text`

### Embedding and Vector Store

Embedding and upsert logic live in [src/core/embed.py](../src/core/embed.py).

The system uses:

* OpenAI `text-embedding-3-small`
* persistent local Chroma storage

Chunks are upserted with stable IDs so repeated ingest runs update existing records instead of creating duplicates.

### Retrieval

Retrieval lives in [src/utils/retriever.py](../src/utils/retriever.py).

The retriever:

* embeds the query
* queries Chroma for the nearest chunks
* returns chunk text and source metadata

This is a dense retrieval setup, not sparse retrieval.

### Generation

Generation lives in [src/core/generate.py](../src/core/generate.py).

The answer generator coordinates:

* retrieval
* LLM reranking
* reflection
* answer generation

Prompts are separated into [src/utils/prompts.py](../src/utils/prompts.py), and lightweight output validation lives in [src/utils/validators.py](../src/utils/validators.py).

The LLM is required to return JSON, and the application validates each response before using it.

### Evaluation

Evaluation lives in [src/core/evaluate.py](../src/core/evaluate.py).

It supports:

* full eval-set runs that write to [results/eval_results.json](../results/eval_results.json)
* request-level background evaluation that updates saved `/ask` request artifacts and daily summaries

## Data Layout

Main runtime data locations:

* [data/catalog/catalog.json](../data/catalog/catalog.json)
* [data/essays](../data/essays)
* [data/chunks/recursive_chunks.json](../data/chunks/recursive_chunks.json)
* `data/chroma/` for the local Chroma store
* [results/eval_results.json](../results/eval_results.json)
* `results/ask/<day>/requests/` for per-request traces
* `results/ask/<day>/summary.json` for daily ask summaries

## Logging

Logging utilities live in [src/utils/logger.py](../src/utils/logger.py).

The app writes:

* colored terminal logs
* daily grouped log files

Logs are split by area, for example:

* system startup
* endpoint calls
* pipeline steps
* evaluation

## Why It Is Still Simple

This is intentionally an MVP. A few places are simpler than a production system on purpose:

* synchronous generation path instead of async orchestration
* simple HTML ingestion instead of a general crawler
* a lightweight LLM reranker instead of a dedicated cross-encoder
* JSON artifacts on disk for transparency and debugging
* a small manual evaluation set rather than a large benchmark

The tradeoff is that the system is easier to inspect and explain, but less optimized and less robust than a more mature implementation would be.

## Known Weak Spots

The main weak spots at the moment are:

* reranking is the least stable part of the pipeline
* retrieval quality is still heavily dependent on chunking quality
* the current evaluation setup is useful but still forgiving
* hallucination mitigation reduces risk, but does not fully eliminate it

These are acceptable tradeoffs for the scope of the take-home, but they are also the first places I would improve next.
