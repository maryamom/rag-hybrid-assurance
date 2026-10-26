# API reference

Base URL: `http://localhost:8005`. Interactive documentation: `/docs`.

Every response sets a `bhbot_session` cookie (HttpOnly) on the first request. Conversation history and
authentication are tied to it: send it back (browsers do so automatically).

## `POST /api/query`

```json
{ "query": "Que couvre la garantie bris de glace ?" }
```

Response:

```json
{
  "response": "La garantie bris de glace couvre le pare-brise et les vitres latérales [1].",
  "intent": "general",
  "sources": [{ "title": "Bris de glace", "kind": "garantie", "score": 0.0325 }],
  "quotation": null,
  "needs_auth": false,
  "auth_step": null,
  "vector_results_count": 3,
  "graph_results_count": 2,
  "processing_time": 1.84,
  "timestamp": "2026-10-26T21:30:54.697754+00:00"
}
```

`intent` is `general`, `quotation`, `personal_data` or `authentication`. During authentication,
`needs_auth` is `true` and `auth_step` is `identity` or `code`.

## `POST /api/quotation`

Quote from the form, all fields validated:

```json
{
  "n_cin": "12345678",
  "valeur_venale": 45000,
  "valeur_a_neuf": 52000,
  "date_premiere_mise_en_circulation": "2019-01-01",
  "puissance": 6,
  "nombre_place": 5,
  "classe": 3,
  "nature_contrat": "r",
  "capital_bris_de_glace": 900,
  "capital_dommage_collision": 60000
}
```

`200` → `{ "error": false, "quotation": { ... } }` · `422` invalid fields · `502` quote API unavailable.

## `POST /api/session/reset`

Forgets this browser's conversation and signs the client out.

## `GET /api/health`

```json
{ "status": "healthy", "services": { "qdrant": true, "neo4j": true, "postgres": true, "llm": true } }
```

`status` is `healthy` (all up), `degraded` (at least one search source up) or `down`.
