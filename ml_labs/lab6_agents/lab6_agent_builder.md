# Lab 6: Vertex AI Agent Builder — RAG-Based Study Assistant

**Status:** ✅ Complete  
**Duration:** ~3 hours  
**Cost:** < $2 (datastore indexing + query costs)

---

## Objective

Build a RAG-based agent using Vertex AI Agent Builder that answers questions from personal certification notes and official GCP documentation. This directly covers the exam pattern: "build a self-help tool using internal docs with minimal maintenance."

---

## What We Built

A study assistant agent grounded in 42 documents — personal lab notes from Labs 1-5, a multi-agent research pipeline codebase, ML Crash Course notes, and official Google Cloud Vertex AI documentation.

**Architecture:**

```
User question
    ↓
Agent (Gemini 3.0 Flash + system instructions)
    ↓
Vertex AI Search (retrieval from datastore)
    ↓
ml-certification-kb datastore
    ↓
42 indexed documents in GCS
    (parsing → chunking → embedding → vector index)
```

---

## Part 1: Prepare Knowledge Base

### Documents Collected (~42 files)

**Personal materials:**
- Certification plan markdown
- Lab notebooks from Labs 1-5 (BigQuery ML, Vertex AI Pipeline, Hyperparameter Tuning, Monitoring, MLOps Services)
- ML Crash Course notes from Obsidian (text-only — screenshots excluded)
- Multi-agent research pipeline prompt files (10 agent prompts)

**Official GCP documentation (downloaded as .txt):**
- Vertex AI overview
- Vertex AI for BigQuery users
- Hyperparameter tuning overview + creating tuning jobs
- Feature Store overview
- Agent Builder overview
- Grounding overview + grounding with Vertex AI Search
- Creating search data stores
- About apps and data stores

### Key Decision: File Format

Agent Builder does **not** accept `.md` (text/markdown) files. The supported MIME types include `text/plain`, `application/pdf`, `text/html`, and various Office formats — but not markdown.

**Fix:** Rename `.md` → `.txt` before uploading. Content is identical; only the extension changes.

```bash
find . -name "*.md" -exec bash -c 'mv "$1" "${1%.md}.txt"' _ {} \;
```

### Upload to GCS

All files uploaded to: `gs://carty-470812-ml-census-data/agent-builder/knowledge-base/`

Organised into subdirectories:
- `gcp-docs/` — downloaded official documentation
- `my-notes/` — personal lab notes, certification plan, agent prompts

---

## Part 2: Create Datastore

**Console path:** Agent Builder → Data Stores → Create Data Store

**Configuration:**
| Setting | Value |
|---------|-------|
| Source | Cloud Storage |
| GCS path | `gs://carty-470812-ml-census-data/agent-builder/knowledge-base/` |
| Data type | Unstructured documents |
| Parser | Layout parser |
| Name | `ml-certification-kb` |
| Location | Global |
| Datastore ID | `ml-certification-kb_1771757393829` |

**Layout parser** was chosen over the default parser because it understands document structure (headings, sections, lists) rather than treating everything as flat text. This improves chunk quality for structured markdown/text files.

### What Happens During Indexing

The datastore creation triggers a RAG pipeline:

1. **Parsing** — layout parser extracts text, preserving structural cues
2. **Chunking** — documents split into smaller passages for retrieval
3. **Embedding** — each chunk converted to a vector using Google's embedding model
4. **Indexing** — vectors stored in an index for fast similarity search

This is all managed — no configuration of embedding models, chunk sizes, or vector databases required.

---

## Part 3: Test with Vertex AI Studio (Grounding)

Before building the agent, tested retrieval quality using Vertex AI Studio with grounding enabled.

**Console path:** Vertex AI Studio → Freeform → Enable Grounding → Customize → Vertex AI Search

**Datastore path format:**
```
projects/carty-470812/locations/global/collections/default_collection/dataStores/ml-certification-kb_1771757393829
```

### Test Results

**Q: "Which agent in the research pipeline handles technology stack discovery?"**
- ✅ Correctly identified Agent 4 (Tech Stack Discovery)
- ✅ Listed all 6 agents with correct descriptions
- ✅ Cited sources: `1_agent_industry`, `2_agent_business`, `3_agent_people`, `4_agent_tech`, `5_agent_synthesis`
- Response time: ~7 seconds

**Q: "What is Bayesian optimization and how does Vertex AI use it for hyperparameter tuning?"**
- ✅ Comprehensive explanation combining official docs
- ✅ Covered surrogate models, acquisition functions, parallel trials, transfer learning
- ✅ Cited sources: `hyperparameter-tuning-overview`, `vertex-ai-overview`
- Response time: ~11 seconds

**Q: "What accuracy did the custom model achieve vs AutoML in Lab 2?"**
- ✅ Exact numbers: Custom 87.10% vs AutoML 86.8%
- ✅ Pulled cost comparison: $0.04 vs $10-15
- ✅ Pulled training time: 12 min vs 2+ hours
- ✅ Cited sources: `lab2_custom_training`, `lab2_vertex_ai_pipeline`
- ⚠️ Hallucination: Added plausible but incorrect reasoning about WHY the custom model won (attributed it to specific hyperparameters not in the source docs)

**Q: "When should I use Feature Store vs the TRANSFORM pattern in BigQuery ML?"**
- ✅ Successfully combined THREE sources: Feature Store doc, Lab 1 notes, Lab 5 notes
- ✅ Produced accurate decision framework
- ✅ Correctly identified: single model/SQL → TRANSFORM; multiple models/teams/low-latency → Feature Store
- This was the hardest test — required cross-source retrieval and synthesis

### Key Observation: Hallucination with Grounding

Grounding **reduces but does not eliminate** hallucination. The retrieval step was consistently accurate — correct documents, correct chunks. But the generation step sometimes filled narrative gaps with plausible-sounding details not in the source material. Citations allow users to verify claims against actual sources.

---

## Part 4: Build the Agent

**Console path:** Vertex AI → Agent Builder → Agent Designer

**Configuration:**
| Setting | Value |
|---------|-------|
| Name | `ml-cert-study-assistant` |
| Model | Gemini 3.0 Flash |
| Tools | Google Search, Vertex AI Search (ml-certification-kb datastore) |

**System Instructions:**
```
You are a study assistant for the Google Cloud Professional Machine Learning Engineer certification.

Your role:
- Answer questions using ONLY the provided knowledge base documents
- Cite which document your answer comes from
- If the answer is not in the documents, say "I don't have this in my knowledge base" rather than guessing
- When explaining concepts, connect theory to practical examples from the lab notes where possible
- If a question spans multiple topics, structure your answer clearly

Your knowledge base contains:
- Personal lab notes from hands-on GCP ML labs (Labs 1-5)
- A multi-agent research pipeline codebase
- Official Google Cloud documentation for Vertex AI services
- ML certification study plan and notes

Keep answers concise but thorough. Use the lab results as concrete examples when relevant.
```

### Agent vs Raw Grounding

The system instructions changed the response style noticeably. The agent produced more structured, tutor-like responses that connected theory to lab examples — versus Vertex AI Studio which gave more generic search-engine-style answers.

### Generated Code (ADK)

The "Get code" button generated Agent Development Kit (ADK) Python code showing the underlying architecture:

- A **root agent** with system instructions
- **Three sub-agents as tools**: Google Search, URL Context, Vertex AI Search
- The root agent delegates to the appropriate sub-agent based on the query

This is a multi-agent orchestration pattern — simpler than but architecturally similar to the custom sales research pipeline I built separately.

---

## Part 5: Architecture & Exam Patterns

### Agent Builder Components

The Agent Builder section of the console contains several services at different abstraction levels:

| Service | What It Does | Control Level | Use When |
|---------|-------------|---------------|----------|
| **Vertex AI Search** | Fully managed RAG — handles parsing, chunking, embedding, indexing, retrieval | Lowest (most managed) | "Q&A over docs with minimal effort" |
| **RAG Engine** | Configurable RAG — choose embedding model, chunk sizes, retrieval strategy | Medium | Default search behaviour isn't good enough |
| **Vector Search** | Pure vector database — bring your own embeddings | Highest | Custom RAG pipeline, full control needed |

**The spectrum:**
```
More managed ────────────────────────────── More control
Vertex AI Search → RAG Engine → Vector Search → Custom code
(this lab)                                       (sales pipeline)
```

### Decision Framework for Exam Questions

| Scenario | Answer |
|----------|--------|
| Q&A over existing docs, minimal maintenance | **Agent Builder + datastore** |
| Model needs to learn new behaviour/style | **Fine-tuning** |
| Custom retrieval logic or non-standard pipeline | **Custom RAG on GKE with Vector Search** |
| Simple keyword search over documents | **Vertex AI Search (no agent)** |
| "Minimize code/effort" + "internal documentation" | **Agent Builder** (always) |

### Exam Signal Words

- "internal documentation" → grounding/RAG, not fine-tuning
- "minimize maintenance" → managed service (Agent Builder), not GKE
- "build quickly" → Agent Builder, not custom pipeline
- "custom retrieval logic" → RAG Engine or Vector Search
- "learn new style/behaviour" → fine-tuning

### Grounding vs Fine-Tuning

| | Grounding (RAG) | Fine-tuning |
|--|-----------------|-------------|
| **How** | Retrieve relevant docs at query time, inject into prompt | Modify model weights with training data |
| **When** | Model needs access to specific documents/data | Model needs to learn new behaviour, style, or domain knowledge |
| **Data changes** | Just update the datastore — no retraining | Must retrain the model |
| **Cost** | Pay per query (retrieval + generation) | Pay for training + serving |
| **Latency** | Slightly higher (retrieval step) | Same as base model |
| **Hallucination** | Reduced but not eliminated | Can still hallucinate |

---

## Key Learnings

1. **Agent Builder is the managed RAG solution.** Upload docs, create datastore, connect to agent — done. No embedding model selection, no chunk size tuning, no vector database management.

2. **Markdown files must be renamed to .txt.** Agent Builder doesn't recognise `text/markdown` as a valid MIME type. Same content, different extension.

3. **Layout parser > default parser** for structured documents. It preserves headings, sections, and hierarchy which improves chunk quality.

4. **Grounding reduces but doesn't eliminate hallucination.** Retrieval was consistently accurate, but generation sometimes filled gaps with plausible but incorrect details. Citations are the mitigation.

5. **System instructions change response quality significantly.** The agent produced more useful, contextual responses than raw grounded search in Vertex AI Studio.

6. **Cross-source retrieval works.** The Feature Store vs TRANSFORM question successfully combined three different source documents into a coherent answer.

7. **Agent Builder generates ADK code.** The console is a no-code frontend for the Agent Development Kit — you can export and customise the generated code.

8. **Three levels of RAG in GCP:** Vertex AI Search (managed) → RAG Engine (configurable) → Vector Search (DIY). Exam tests whether you know which to pick for a given scenario.

9. **The `default_collection` is the default Collection ID** for all datastores unless explicitly customised.

10. **Datastore path format matters:**
    ```
    projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATASTORE_ID}
    ```

---

## Cost Breakdown

| Item | Cost |
|------|------|
| GCS storage (42 files, ~1.3 MB) | < $0.01 |
| Datastore indexing | ~$0.50 |
| Query testing (~20 queries) | ~$0.50 |
| Agent preview testing | ~$0.50 |
| **Total** | **< $2.00** |



## Bonus: RAG Engine vs Vertex AI Search Comparison

After completing the main lab with Vertex AI Search, tested the same knowledge base through RAG Engine to compare retrieval quality.

### Setup Difference

| Setting | Vertex AI Search (Datastore) | RAG Engine (Corpus) |
|---------|------------------------------|---------------------|
| Console path | Agent Builder → Data Stores | Vertex AI → RAG Engine |
| Chunk size | ~500 tokens (default, not configurable) | 1024 tokens (configurable) |
| Embedding model | Google default (not configurable) | Configurable |
| Chunking strategy | Managed | Configurable overlap and size |

### Test: "I'm confused about when to use AutoML vs custom training. Can you explain using examples from my labs?"

**Vertex AI Search response:**
- ✅ Pulled correct Lab 2 numbers (87.10% vs 86.8%, $0.04 vs $10-15)
- ❌ Hallucinated incorrect lab descriptions: claimed Lab 2 was "predict customer purchasing" and Lab 3 was "image classification"
- Used 2 source documents

**RAG Engine response:**
- ✅ Pulled correct Lab 2 numbers with accurate context
- ✅ No hallucinated lab details — stuck to what was in the documents
- ✅ Found an additional relevant source: ML Crash Course notes (`4b - Automated Machine Learning.txt`) for the "when to use AutoML" rationale
- ✅ Cleaner structure with specific lab evidence for each recommendation
- Used 3 source documents

### Why the Difference?

The larger chunk size (1024 vs ~500 tokens) likely explains the improvement. Bigger chunks preserve more context around each fact, so when the retrieval step finds a chunk about Lab 2 results, it includes enough surrounding text to understand the full picture — not just a fragment. This reduces the need for the generation step to fill gaps, which is where hallucination happens.

The tradeoff: larger chunks are less precise — you might pull irrelevant content alongside the relevant bit. For structured documents (lab notes with clear sections), larger chunks work well. For massive unstructured documents, smaller chunks with more precise retrieval might be better.

### Exam Relevance

This demonstrates why the three-tier RAG architecture exists:

- **Vertex AI Search**: Works out of the box, no tuning needed. Pick this when the default is good enough.
- **RAG Engine**: Tune chunk size, overlap, embedding model. Pick this when retrieval quality needs improvement.
- **Vector Search**: Full control over everything. Pick this when you need custom embeddings or non-standard retrieval logic.

If an exam question mentions "retrieval quality isn't sufficient with the default configuration," RAG Engine is the answer.

---

## Comparison: Agent Builder vs Custom Pipeline

Having built both a managed Agent Builder agent and a custom multi-agent research pipeline, the key differences are:

| Dimension | Agent Builder | Custom Pipeline (Gemini API) |
|-----------|--------------|------------------------------|
| Setup time | ~30 minutes | Days-weeks |
| Control over prompts | System instructions only | Full prompt engineering per agent |
| Retrieval | Managed (Vertex AI Search) | Google Search grounding or custom |
| Output format | Text/chat | Structured JSON → PPTX, Excel |
| Multi-agent | ADK sub-agents (simple delegation) | Custom orchestration (parallel, sequential, context passing) |
| Cost per query | Higher (managed infra) | Lower (direct API calls) |
| Maintenance | Google manages everything | You manage everything |
| Best for | Internal tools, Q&A, prototypes | Production workflows, structured outputs |

---

## Files

- `upload_knowledge_base.sh` — Script to download GCP docs and upload knowledge base to GCS