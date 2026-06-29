"""Cypher safety helpers for the LLM-generated read path.

Two helpers used by the neo4j_retriever_node:

- `is_cypher_read_only(cypher)` rejects any Cypher that contains write
  keywords (CREATE, MERGE, DELETE, DROP, SET, REMOVE). The retriever runs
  this BEFORE sending to Neo4j; rejected queries skip execution entirely.
- `wrap_with_limit(cypher, limit)` wraps the LLM's Cypher in a CALL { ... }
  subquery with a database-enforced outer LIMIT. Even if the LLM omits
  LIMIT (or uses a high one), the server caps the row count.

Why two layers, not one:
- The validator catches the dangerous keywords up front - cheaper than
  letting Neo4j parse + reject them, and safer than relying on a read-only
  session alone (session-level read-only is the third belt; see
  `Neo4jClient.read_query`).
- The wrapper handles the volume-control problem orthogonally to safety.

Limitations:
- The keyword check is regex-based (word-boundary, case-insensitive).
  A forbidden keyword inside a string literal or comment also rejects -
  defensive false positive. LLM output rarely contains either.
- The wrapper assumes the inner Cypher is a self-contained read query
  ending with RETURN. Exotic Cypher (multi-statement scripts, USING INDEX
  hints) may not nest. The retriever's fail-soft handler catches the
  resulting exception.
"""

import re

FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DROP",
    "SET",
    "REMOVE",
)

# Word-boundary regex for the forbidden keywords. Compiled once at module
# load. Case-insensitive so "create", "Create", "CREATE" all match.
_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_cypher_read_only(cypher: str) -> bool:
    """Return True iff the Cypher contains no write keywords.

    Word-boundary match: a property name like `created_at` does NOT trigger
    a rejection (no boundary between `created` and `_at`), but `CREATE`,
    `Create`, and `create` (as standalone tokens) all do.
    """
    return _FORBIDDEN_PATTERN.search(cypher) is None


def wrap_with_limit(cypher: str, limit: int) -> str:
    """Wrap a read Cypher query in a CALL subquery with a final LIMIT.

    The outer LIMIT is enforced by Neo4j itself, so this caps the rows
    returned regardless of whether the inner query has its own LIMIT
    clause. The caller is responsible for passing a sensible `limit`
    (typically `settings.kg_max_rows`).
    """
    return f"CALL {{\n{cypher}\n}}\nRETURN * LIMIT {limit}"
