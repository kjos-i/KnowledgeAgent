# Searching

Ask a question of the selected corpus in the chat on the left; the answer appears
in the right column with citations. The **Retrieval** and **LLMs** sub-tabs (this
same column) tune how the search runs and which model answers.

Jump to: [Query modes](#query-modes) · [Retrieval modes](#retrieval-modes) ·
[Within-store search](#within-store) · [The knobs](#knobs) · [Cypher tips](#cypher)

<a id="query-modes"></a>
## Query modes: how your text becomes a query

Set on **Retrieval**. This decides how your typed text is interpreted:

| Mode | What it does |
|---|---|
| **Refined query** | A query-builder rewrites your text before searching. |
| **Direct query** | Searches your exact text, unrewritten (skips the query builder). |
| **Direct Cypher** | Runs your text as raw Cypher on the knowledge graph, forcing the store to graph-only. |

The chat's **Conversational** mode (the default) is a chat-only pre-step: the chat
router reads the whole conversation and either asks a clarifying question or distils
a standalone query and searches it. When it searches, it uses the Refined path (the
query builder runs) with your selected retrieval mode.

<a id="retrieval-modes"></a>
## Retrieval modes: which stores run

Picks which store(s) answer, and in what order. The **Path** column traces a query
through the stores (LanceDB = the vector store, Neo4j = the knowledge graph):

| Mode | Path |
|---|---|
| **auto** | picks one of the modes below, per query |
| **lancedb_only** | query → LanceDB → answer |
| **neo4j_only** | query → Neo4j → answer |
| **lancedb_then_neo4j** | query → LanceDB → Neo4j → answer |
| **neo4j_then_lancedb** | query → Neo4j → LanceDB → answer |
| **parallel_fused** | query → LanceDB + Neo4j → fuse → answer |

<a id="within-store"></a>
## Within-store search (LanceDB)

When the LanceDB leg runs, it can search three ways:

| Mode | How |
|---|---|
| **Hybrid** | BM25 keyword + vector similarity, fused with RRF (the default). |
| **FTS** | BM25 full-text only: keywords, no vectors. |
| **Vector** | kNN cosine similarity only: semantic, no keywords. |

<a id="knobs"></a>
## The knobs: which modes use them

| Knob | Default | Applies when |
|---|---|---|
| **top_k** | 5 | Always: how many results to return. |
| **num_candidates** | 100 | Hybrid or Vector search (not FTS), in any mode that runs LanceDB. |
| **rrf_rank_constant** | 60 | Hybrid only: it's the fusion constant. |
| **MMR / mmr_lambda** | 0.6 | When MMR is enabled, on Hybrid or Vector search (FTS has no vectors). λ = 1 → pure relevance, λ = 0 → pure diversity. |
| **kg_max_rows** | 50 | Any mode that runs Neo4j (not `lancedb_only`). |

The form grays out a knob the current mode can't use, and the search never reads
a grayed knob: either the store leg that would read it doesn't run, or the LanceDB
search mode in use doesn't need it (FTS uses none of num_candidates, RRF, or MMR).

<a id="cypher"></a>
## Cypher tips

**Direct Cypher** lets you query the knowledge graph yourself instead of letting
the query-builder do it. Pick **Direct Cypher** on Retrieval and type a Cypher
query in the chat; it runs on the graph (the store is pinned to graph-only).

- The graph holds your documents' entities and relationships. Get your bearings
  with a broad query first, e.g.
  `MATCH (n) RETURN labels(n), count(*) ORDER BY count(*) DESC`.
- Always `RETURN` something: a bare `MATCH` shows nothing.
- Keep it to a plain read: the runtime rejects any write (`CREATE`, `MERGE`,
  `DELETE`, `DROP`, `SET`, `REMOVE`), the `SHOW` and `LOAD` commands, and every
  procedure call (`CALL` is blocked outright, including read-only ones like
  `db.labels()`, and the whole `apoc.` / `dbms.` namespace). A rejected query
  returns nothing, so stick to `MATCH … RETURN …`. Ingestion is what writes the
  graph.
