# Architecture

## Request path

`POST /api/query` → `HybridRAGOrchestrator.answer(session_id, question)` (`src/core/orchestrator.py`).

| Order | Condition | Handler | Data read |
| --- | --- | --- | --- |
| 1 | an authentication is in progress for this session | `_continue_auth`: identity, then code | PostgreSQL `clients` |
| 2 | the question is about the user's own data (« mes contrats », « ma prime »…) | `_personal_data`: authenticate, then answer from the client's records | PostgreSQL `clients`, `devis`, `contrats`, filtered on the client's id |
| 3 | a quote request, or a quote in progress | `_quotation`: extract, ask for missing fields, validate, call the API | quote API |
| 4 | anything else | `_rag`: rephrase with history, hybrid retrieval, grounded answer | Qdrant + Neo4j (public data only) |

Routing (`src/core/detectors.py`) is keyword-based and runs before any LLM call.

## Retrieval

- **Qdrant** (`retrievers/vector.py`): one similarity search, filtered on `source_type ∈ PUBLIC_VECTOR_TYPES`
  (`document`, `garantie`, `produit`, `faq`, `dictionnaire`). Multilingual embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2` by default).
- **Neo4j** (`retrievers/graph.py`): one query on the `public_knowledge` full-text index (French analyzer), restricted
  to `PUBLIC_GRAPH_LABELS`, plus up to 6 public neighbours per node. The user's question is escaped so Lucene
  syntax can't be injected.
- **Fusion** (`retrievers/hybrid.py`): both sources are queried in parallel with a 15 s timeout each and merged
  with Reciprocal Rank Fusion (k = 60). Identical passages are merged.

The prompt (`src/core/prompts.py`) asks for an answer based only on the numbered passages, with citations, and an
explicit « I don't know » when the passages don't contain the answer.

## Public and private data

| Store | Content | Who can read it |
| --- | --- | --- |
| Qdrant | PDF passages, guarantee descriptions, product/profile mapping, FAQ, data dictionary | anyone |
| Neo4j | Branche, SousBranche, Produit, Garantie, ProfilCible, QuestionReponse | anyone |
| PostgreSQL | clients, quotes, contracts, OCR text | the authenticated client, own records only |

`src/core/graph_schema.py` holds the allow-lists used both by the ingestion scripts and the retrievers.
The knowledge graph is built from distinct product, branch and guarantee names only: no person, contract
number or amount reaches Neo4j.

## Authentication

`src/auth/client_auth.py`:

1. The client gives first name, last name and phone number.
2. If they match a client, a 6-digit code (`secrets.randbelow`) is e-mailed to the address on file. The reply is
   the same whether or not the client exists.
3. The code is stored as a salted SHA-256 hash, bound to the session id, valid 10 minutes, 3 attempts.
4. A valid code creates a client token (1 hour) kept in the session; follow-up questions about « mes contrats »
   don't ask for a new code.

`AUTH_DEMO_MODE=true` prints codes in the server console instead of e-mailing them (local demos only).

## Ingestion

See the table in the [README](../README.md#quick-start). Table and column names of the hackathon files are
declared once in `ingestion/schema.py`; `python -m ingestion.check_data` reports what was found.
