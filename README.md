# rag-search-your-base

A Retrieval-Augmented Generation (RAG) question-answering system over the Paul Graham essays. The pipeline covers ingestion, chunking, dense retrieval, reranking, grounded answer generation, lightweight hallucination mitigation, and evaluation.

I chose the Paul Graham essays because they are freely available, long-form, and varied enough for chunking and retrieval choices to actually matter. They were also unfamiliar enough to me that I could write more honest evaluation questions instead of leaning too much on prior knowledge.

## Setup

Use `requirements.txt` as the pinned dependency source of truth for installation. `pyproject.toml` is there for packaging metadata, but reviewers should install from `requirements.txt`.

Primary development environment: Ubuntu Linux.

API keys are not included in this repository.

To run the project locally:
- Create an OpenAI API key in the OpenAI Platform and set it as OPENAI_API_KEY in .env. OpenAI manages API keys from the Platform settings/project key pages. https://platform.openai.com
- Create a Gemini API key in Google AI Studio and set it as GEMINI_API_KEY in .env. Google’s Gemini docs explain that API keys are created and managed in Google AI Studio. https://aistudio.google.com/api-keys

### OpenAI
1. Sign in to the OpenAI Platform
2. Go to API keys in your project/settings
3. Create a new API key
4. Copy it into .env as OPENAI_API_KEY=...

### Gemini
1. Sign in to Google AI Studio
2. Go to API Keys
3. Create a Gemini API key
4. Copy it into .env as GEMINI_API_KEY=...

### Linux / Ubuntu

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Environment Variables

Set the following in `.env`:

* `OPENAI_API_KEY`
* `GEMINI_API_KEY`
* `GEMINI_JUDGE_MODEL`
* `LOG_LEVEL`
* `CHROMA_COLLECTION_NAME`
* `PAUL_GRAHAM_INDEX_URL`
* `HOST`
* `PORT`

Reviewers should substitute their own API keys without changing the code. OpenAI is paid, but I chose a relatively affordable model. Gemini is being used on the free side.

## How To Run

Start the API from the repo root:

```bash
PYTHONPATH=src python -m main
```

Open:

* `http://127.0.0.1:8000/docs`

Useful endpoints:

* `POST /discover`
* `POST /ingest`
* `POST /ask`
* `GET /health`

Manual evaluation:

```bash
PYTHONPATH=src python -m core.evaluate
```

## Knowledge Base Choice

I used the Paul Graham essays as the knowledge base for this project.

The main reasons were:

* they are freely available
* there is enough material to make retrieval meaningful
* they are written as long-form prose, which makes chunking tradeoffs more visible
* I do not know them well enough to fake a good evaluation set from memory

That made them a good fit for a small but realistic RAG pipeline.

## Ingestion Approach

For ingestion, I used a simple HTML scraping flow over the Paul Graham essays index page.

The ingestion layer:

* fetches the full essay list from the index
* fetches individual essays
* stores cleaned essay text locally as JSON

I kept this part deliberately simple. The point of the exercise is the RAG pipeline, not building a crawler or ingestion platform, so I did not want to spend too much time overengineering this layer.

I used BeautifulSoup because:

* the source site is public and structurally simple
* the code is easy to reason about
* it keeps the ingestion logic easy to inspect and modify
* it produces stable local artifacts for chunking and indexing

### Limitations

This approach assumes the source HTML structure stays reasonably stable. It also only does lightweight cleaning and does not try to handle every formatting edge case.

## Chunking Strategy

I used recursive text chunking with a chunk size of `1000` characters and an overlap of `200` characters.

I went with this because it felt like a reasonable balance between simplicity and retrieval quality:

* it is easy to implement and explain
* it is better than just chopping text at hard character boundaries
* it preserves more local continuity than naive fixed-size chunking
* overlap helps reduce context loss at chunk boundaries

For this corpus, which is mostly essay-style prose rather than highly structured technical documentation, recursive chunking felt like a good baseline. I used LangChain’s `RecursiveCharacterTextSplitter` instead of building a custom chunker so I could spend more time on retrieval, generation, hallucination mitigation, and evaluation.

### Limitations

This strategy does not explicitly preserve paragraph or semantic boundaries, so some chunks will still split ideas less cleanly than a paragraph-aware or semantic chunker would.

## Retrieval

I implemented dense retrieval using OpenAI embeddings and Chroma as the vector store.

* embedding model: `text-embedding-3-small`
* vector database: Chroma persistent local store
* retrieval style: dense vector similarity search

The retriever brings back a bounded set of chunks, and the generation pipeline reranks them before deciding which ones to pass into answer generation. The idea was to keep the prompt smaller while still allowing more than one chunk through when multiple pieces of evidence looked useful.

### Limitations

* dense retrieval quality still depends heavily on the embeddings and chunk boundaries
* the reranking layer is intentionally lightweight and still a bit brittle
* some relevant chunks may still be missed or pushed down

## Generation

Generation uses an OpenAI model to produce the final answer using retrieved context only.

The flow is:

1. retrieve candidate chunks
2. rerank them
3. reflect on whether the selected context is sufficient
4. answer using only the provided context

The answer step returns:

* `answer`
* `citations`
* `confidence_score`

I kept generation synchronous for this project because:

* the current OpenAI client wrapper is synchronous
* retrieval, reranking, and answer generation happen in a short linear path
* keeping it synchronous made the control flow easier to trace in logs
* it kept complexity down while I focused on getting the pipeline working properly end to end

At this stage, simplicity and debuggability mattered more to me than adding async orchestration early. I also was not trying to optimize for speed yet. There are better ways to improve latency later, but I preferred to get the pipeline working and inspectable first.

## Hallucination Mitigation

I implemented hallucination mitigation using source grounding and prompt-level constraints. The model only receives retrieved chunks and is instructed to answer strictly from that context, cite the supporting sources it used, and abstain when the evidence is not good enough.

The main guardrails are:

* outside knowledge is explicitly forbidden at the prompt level
* every non-abstaining answer must include citations from the retrieved context in `(title)[url]` format
* abstention is mandatory when support is insufficient: `"I don't know based on the provided context."`
* the application performs lightweight validation of LLM outputs before using them

This matters because the model may already know some of these essays from pretraining. The goal here was to reduce that risk by forcing the answer to stay tied to retrieved evidence and requiring source-backed output.

### Limitations

This depends heavily on retrieval quality. If the right chunks are not retrieved or reranked well, the model can still produce an incomplete answer or abstain when it probably should not.

## Evaluation

I used `Recall@3` as the primary retrieval metric.

In this project, `Recall@3` means:

* each evaluation question has an `expected_slugs` set
* retrieval returns the top 3 essay sources
* per-question recall is:

```text
relevant retrieved in top-3 / total relevant expected sources
```

I supplemented that with Gemini as an LLM judge. The judge does not replace the metric. It is there to add qualitative context around:

* whether the retrieved sources look appropriate
* whether the generated answer aligns with the expected answer summary
* whether the result should be seen as strong, weak, or ambiguous

### Results Table

Metric: `Recall@3`

`Recall@3 = 6 / 6 = 1.00`

| Question                                                                        | Expected Source | Retrieved Top 3 Sources                        | Recall@3 |
| ------------------------------------------------------------------------------- | --------------- | ---------------------------------------------- | -------- |
| What does Paul Graham say great work depends on besides effort and luck?        | greatwork       | greatwork, greatwork, greatwork                | 1.0      |
| Why does Paul Graham think writing that sounds good is more likely to be right? | goodwriting     | goodwriting, goodwriting, goodwriting          | 1.0      |
| What does Paul Graham say people should do in life?                             | do              | do, do, do                                     | 1.0      |
| Why does Paul Graham say AI will create writes and write-nots?                  | writes          | writes, writes, writes                         | 1.0      |
| What made Paul Graham change his mind about having kids?                        | kids            | kids, kids, kids                               | 1.0      |
| According to Paul Graham, how do people usually lose time and money?            | selfindulgence  | selfindulgence, selfindulgence, selfindulgence | 1.0      |

### Interpretation

The current `Recall@3` result looks strong, but I do not think it should be taken at face value. It is still a forgiving metric on a small corpus, especially when the evaluation questions map fairly directly to one essay and the retriever only needs to get the expected source somewhere into the top 3.

It also does not tell me whether the best chunk was retrieved first, whether the retrieved evidence was strong enough, or whether the final answer was actually correct.

That weakness already showed up in one of the examples: retrieval succeeded, but the answer still did not align particularly well with the expected answer summary. So the current numbers are useful as a basic retrieval signal, but they make the system look stronger than it probably is end to end.

### If I Had More Time

If I had more time, I would:

* simplify and stabilize the reranking layer further
* compare the current chunking baseline against a paragraph-first alternative
* expand the evaluation set with more multi-source questions
* add a few end-to-end integration tests around `/ask`
* tighten the evaluation itself, because the current recall setup probably flatters the retriever

I see the current evaluation as a decent baseline, not something I’d treat as conclusive. The `Recall@3` score was high, which suggests the retriever was generally able to surface the expected essay within the top 3 results. But that metric is still quite forgiving, especially with a small corpus and fairly direct, source-specific questions. It also doesn’t tell me whether the best chunk came back first, or whether the final answer was actually right. My feeling is that the current setup probably looks better than it really is because the corpus is small, most questions map quite neatly to a single essay, and repeated chunks from the same document still make retrieval look strong at the source level. A stronger next step would be to make the eval set more difficult, include more ambiguous and multi-source questions, and add stricter retrieval checks alongside the current recall metric.

## Sample Queries

These examples come from saved request artifacts in `results/ask/20260319/requests/`.

### 1. What does Paul Graham say great work depends on besides effort and luck?

Answer:

> Ability and interest.

Citations:

* `(How to Do Great Work)[https://paulgraham.com/greatwork.html]`

Source artifact: [20260319_211342.json](/home/shadow-realm/Documents/technical-interview-workspace/rag-search-your-base/results/ask/20260319/requests/20260319_211342.json)

### 2. What does Paul Graham say people should do in life?

Answer:

> Paul Graham says people should help people, take care of the world, and make good new things — in other words, try to create useful new things while making sure they do not harm people or the world.

Citations:

* `(What to Do)[https://paulgraham.com/do.html]`

Source artifact: [20260319_211433.json](/home/shadow-realm/Documents/technical-interview-workspace/rag-search-your-base/results/ask/20260319/requests/20260319_211433.json)

### 3. Why does Paul Graham say AI will create writes and write-nots?

Answer:

> Graham says AI removes almost all the pressure to learn to write because it can generate writing for you, so most people will stop practicing the skill. That collapses the middle ground, leaving those who still choose to write and those who do not. He argues this matters because writing is thinking.

Citations:

* `(Writes and Write-Nots)[https://paulgraham.com/writes.html]`

Source artifact: [20260319_211544.json](/home/shadow-realm/Documents/technical-interview-workspace/rag-search-your-base/results/ask/20260319/requests/20260319_211544.json)

## Known Limitations

* scraping assumes the Paul Graham HTML structure remains stable
* chunking does not explicitly preserve paragraph or semantic boundaries
* dense retrieval is only as good as the embedding and chunking choices
* reranking is intentionally lightweight and has been the least stable part of the pipeline
* hallucination mitigation reduces risk but does not eliminate it
* the evaluation set is small and manually created
* Windows instructions are included, but development was done primarily on Ubuntu

## What I Still Need To Complete

The main optional follow-up item is sanity-checking the Windows setup flow on a real Windows machine, since development was done primarily on Ubuntu.
