# Augy IDP MVP — ADK Edition

A zero-cost, multimodal Intelligent Document Processing (IDP) system built on **Google ADK (Agent Development Kit)**. Ingests PDFs, Word docs, slide decks, spreadsheets, images, audio, and video, stores extracted knowledge across a vector store and a knowledge graph, and answers natural-language questions with full source attribution.

Every answer cites exactly which file, page, or timestamp it came from — and clearly labels anything pulled from the model's own training data instead of your documents.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [How Source Attribution Works](#how-source-attribution-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Extending the System](#extending-the-system)
- [Migrating from the LangGraph Version](#migrating-from-the-langgraph-version)
- [License](#license)

---

## Key Features

- **Multimodal ingestion** — PDF, DOCX, PPTX, CSV, XLSX, PNG, JPG, MP3, WAV, MP4, MOV
- **Deduplication** — MD5 hash manifest skips files already processed
- **Offline transcription** — OpenAI Whisper runs locally, no API key required
- **Vision-based description** — Gemini analyzes images, diagrams, and video frames
- **Knowledge graph** — entity-relationship triples extracted and stored in Neo4j
- **Parent-child vector retrieval** — fine-grained search with full-context fallback in ChromaDB
- **LLM-routed querying** — a Google ADK Agent decides tool order at query time, constrained to always retrieve before answering
- **Source-cited answers** — every response includes inline `[1][2]` citations and a `## Sources used` section
- **Dead-letter queue** — failed files are tracked for inspection, never silently dropped
- **Drive-backed persistence** — manifest and vector store survive Colab session resets

---

## Architecture

```
                         INGESTION PATH (deterministic)
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────────┐
│  Input   │ -> │   Dedup    │ -> │ Format Router│ -> │  Parser Tool   │
│  files   │    │  (MD5)     │    │ (MIME detect)│    │ (Docling/      │
└──────────┘    └────────────┘    └──────────────┘    │  Whisper/cv2/  │
                                                        │  Gemini Vision)│
                                                        └───────┬────────┘
                                                                │
                                                                v
                                                     ┌─────────────────────┐
                                                     │  extract_and_index   │
                                                     │  ChromaDB + Neo4j     │
                                                     └─────────────────────┘

                         QUERY PATH (LLM-routed via ADK)
┌──────────┐    ┌─────────────────────────────────────────────────────┐
│ ask(q)   │ -> │  ADK Runner  ->  query_agent (Gemini 2.5 Flash)        │
└──────────┘    │     decides to call, in order:                        │
                │     1. search_vector_store(q)                         │
                │     2. search_graph_store(q)                          │
                │     3. generate_cited_answer(q, passages, graph_paths)│
                └─────────────────────┬───────────────────────────────┘
                                       v
                          Cited answer + Sources used section
```

The ingestion path is intentionally **not** LLM-routed — file parsing has one correct order every time, so tool functions are called directly in Python. The query path **is** LLM-routed through an ADK `Agent` and `Runner`, because deciding what's relevant genuinely benefits from reasoning.

---

## Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| Orchestration | Google ADK (Agent + Runner) | Free, open source |
| LLM / Vision | Gemini 2.5 Flash | Free tier |
| Audio transcription | OpenAI Whisper (offline) | Free |
| Document parsing | Docling | Free, open source |
| Vector store | ChromaDB + all-MiniLM-L6-v2 | Free, local |
| Graph store | Neo4j Aura | Free tier (50K nodes) |
| Query cache | diskcache | Free, local disk |
| Runtime | Google Colab (T4 GPU) | Free tier |
| Persistence | Google Drive | Free |

---

## Prerequisites

- A Google account with access to Colab and Drive
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)
- *(Optional)* A free Neo4j Aura instance from [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura) for knowledge graph features
- *(Optional)* A Hugging Face token to silence rate-limit warnings on the embedding model download

---

## Setup

### 1. Open the notebook in Google Colab

Upload `Augy_IDP_MVP_ADK.ipynb` to Colab, or open it directly from Drive.

### 2. Run Cell 1 — Installation

Installs `google-adk`, `google-genai`, Docling, ChromaDB, Neo4j driver, Whisper, OpenCV, and `nest_asyncio` (required for ADK's async Runner to work inside Colab's existing event loop).

```python
!pip install -q google-adk google-genai docling chromadb neo4j diskcache \
    openai-whisper opencv-python-headless filetype jsonlines openpyxl \
    pandas Pillow python-dotenv sqlalchemy nest_asyncio
```

### 3. Run Cell 2 — Secrets

On first run, this creates a `.env` file at `/MyDrive/idp_adk/.env` and stops. Open it on Drive and fill in:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

> **Important:** ADK auto-discovers the key under the name `GOOGLE_API_KEY` specifically — not `GEMINI_API_KEY`. If you're migrating secrets from an older version of this project, rename the key.

Re-run Cell 2 after saving.

### 4. Run Cells 3–10 in order

Each cell prints a `✅` confirmation when ready. Cell 8 initializes the ADK session — watch for `[ADK] ✅ Session initialised for query_agent.` before proceeding.

### 5. Drop files into your input folder

```
/content/drive/MyDrive/idp_adk/input/
```

---

## Usage

### Start the ingestion worker

```python
start()
```

Boots a background daemon thread and restores any previous ChromaDB backup from Drive.

### Ingest files

```python
ingest("/content/drive/MyDrive/idp_adk/input/manual.pdf")   # single file
ingest("/content/drive/MyDrive/idp_adk/input/")              # whole folder
ingest("/content/drive/MyDrive/idp_adk/input/*.pdf")         # glob pattern
```

Wait for this exact message before querying:

```
[Worker] ✅ Queue empty. Manifest saved. ChromaDB backed up.
```

### Verify before querying

```python
show_manifest()
child_col, _ = get_chroma_collections()
print(child_col.count())   # must be > 0
```

### Ask a question

```python
ask("How does compressed air flow through the brake cylinder?")
```

This runs through the ADK `query_agent`, which calls `search_vector_store`, then `search_graph_store`, then `generate_cited_answer`, and returns the final cited text.

### Utility commands

| Command | Description |
|---|---|
| `show_manifest()` | Lists all ingested files with chunk counts |
| `show_dead_letters()` | Lists files that failed ingestion after retries |
| `show_log(n)` | Tails the last `n` structured log entries |
| `reset_file('name')` | Removes a file from the manifest to force re-ingestion |

---

## How Source Attribution Works

Every retrieved passage is formatted into a numbered reference before being sent to the model:

```
[1] brake_manual.pdf | page 4 | document
[2] brake_diagram.png | image_3 | image
[3] training_video.mp4 | frame@02:30 | video_frame
```

The `query_agent`'s system instruction mandates:

1. Cite retrieved passages inline as `[1]` or `[2][3]`
2. Label any statement from the model's own training data as `[LLM knowledge]`
3. End every response with a `## Sources used` section
4. If no context was retrieved at all, warn explicitly before answering

This guarantee is enforced even though the LLM is choosing the tool sequence itself — the instruction requires `generate_cited_answer` to run last and its `answer` field to be returned verbatim, so citation formatting can't be skipped.

---

## Project Structure

| Cell | Purpose |
|---|---|
| 1 — Installation | All dependencies, including `google-adk` and `nest_asyncio` |
| 2 — Secrets & Env | Drive mount, `.env` loading, workspace directory setup |
| 3 — Imports | ADK `Agent`/`Workflow` imports, Whisper load, ChromaDB embedding pre-warm |
| 4 — Helpers | Validation gatekeeper, structured JSONL logger, rate-limited `llm_call()` |
| 5 — Ingestion Tools | `parse_document`, `parse_tabular`, `parse_audio`, `parse_video`, `parse_image`, `extract_and_index` |
| 6 — Query Tools | `search_vector_store`, `search_graph_store`, `generate_cited_answer` |
| 7 — Agents | `query_agent` (LLM-routed) and `ingestion_agent` (tool container) |
| 8 — Runner + Worker | `InMemoryRunner`, ADK session setup, format router, background worker |
| 9 — `ask()` | Runs `query_agent` via `adk_runner.run_async()`, extracts final text |
| 10 — Public API | `start()`, `ingest()`, `show_manifest()`, `show_dead_letters()`, `reset_file()` |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `event loop already running` | Confirm `nest_asyncio.apply()` ran in Cell 1 before any Runner call |
| ADK can't find the API key | `.env` must use `GOOGLE_API_KEY`, not `GEMINI_API_KEY` |
| `Session not found` | Re-run Cell 8 to recreate the ADK session |
| `query_agent` skips a tool | Strengthen the instruction text in Cell 7 with explicit "you MUST" language |
| `ImportError` on `Workflow` | You're on ADK 1.x — this is handled automatically; `ingestion_workflow` falls back to `None` |
| `429 RESOURCE_EXHAUSTED` | `llm_call()`'s 2-second floor reduces collisions but can't bypass a hard daily quota — wait for reset |
| "No relevant context retrieved" | Run `child_col.count()` — if `0`, ingestion hasn't finished. Wait for the "Queue empty" message |
| Neo4j DNS / syntax errors | Aura sandboxes expire — create a fresh instance and update `.env`; ensure Cypher uses `IS NOT NULL`, not the deprecated `exists()` |

---

## Extending the System

**Add a new tool to `query_agent`** — write a typed Python function with a clear docstring, add it to the `tools` list in Cell 7, and update the instruction text if it has a required call position.

**Move ingestion to ADK 2.0's `Workflow` class** — if `ADK_WORKFLOW_AVAILABLE` is `True`, a basic `ingestion_workflow` is already scaffolded in Cell 7. Replace `route_and_parse()`'s direct calls with `Workflow` edges to fully graph-orchestrate ingestion.

**Deploy beyond Colab** — swap `InMemoryRunner` for `VertexAiSessionService`, deploy `query_agent` to Cloud Run via the `adk deploy` CLI, and replace the `queue.Queue` worker with a Cloud Function triggered by Cloud Storage uploads.

---

## Migrating from the LangGraph Version

| LangGraph Concept | ADK Equivalent |
|---|---|
| `StateGraph` | `Agent` + `Runner` (or `Workflow` in ADK 2.0) |
| Node function (state in/out) | Tool function (typed args, dict return) |
| `add_edge()` wiring | Agent instruction text |
| `.compile()` | `InMemoryRunner(agent=...)` |
| `.invoke(state_dict)` | `runner.run_async(...)` + iterate `Event`s |
| `TypedDict` state | ADK Session (managed internally) |

See the full **ADK Edition Documentation** for a detailed architectural walkthrough, troubleshooting guide, and extension patterns.

---

## License

Internal MVP project. Not licensed for external distribution.
