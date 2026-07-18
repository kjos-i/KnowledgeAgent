"""Cypher safety helpers for the LLM-generated read path.

Two helpers used by the neo4j_retriever_node:

- `is_cypher_read_only(cypher)` rejects any Cypher containing a write
  keyword (CREATE, MERGE, DELETE, DROP, SET, REMOVE), a procedure / bulk-load
  keyword (CALL, LOAD), or the apoc/dbms namespace (inline functions like
  `apoc.cypher.*` that execute arbitrary Cypher). The retriever runs this
  BEFORE sending to Neo4j; rejected queries skip execution entirely. This is
  what blocks the apoc.load.* / LOAD CSV SSRF + local-file-read + cred-exfil
  vectors that a read-only session alone does NOT stop.
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
  String literals are blanked before the scan, so a forbidden keyword that
  appears only as DATA (e.g. an entity named "gene set" — the "set" is not
  a write clause) no longer false-rejects. A keyword inside a // or /* */
  comment still rejects - a defensive false positive, but LLM read output
  rarely contains comments.
- The wrapper assumes the inner Cypher is a self-contained read query
  ending with RETURN. Exotic Cypher (multi-statement scripts, USING INDEX
  hints) may not nest. The retriever's fail-soft handler catches the
  resulting exception.
"""

import re

FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    # Write clauses.
    "CREATE",
    "MERGE",
    "DELETE",
    "DROP",
    "SET",
    "REMOVE",
    # Procedure calls + bulk load. CALL reaches ANY procedure (apoc.load.*
    # for SSRF / local-file read, apoc.* for arbitrary-Cypher exec, dbms.* /
    # db.* for admin + config); LOAD (CSV) pulls remote/local files. A
    # read-only session does NOT block these. Validation runs on the RAW LLM
    # Cypher, BEFORE wrap_with_limit adds its own outer CALL { ... }, so
    # blocking CALL here is safe; the cypher-builder emits plain
    # MATCH...RETURN graph reads (text/vector search is LanceDB's job).
    "CALL",
    "LOAD",
)

# Word-boundary regex for the forbidden keywords. Compiled once at module
# load. Case-insensitive so "create", "Create", "CREATE" all match.
_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# apoc / dbms FUNCTIONS (e.g. apoc.cypher.runFirstColumn, which executes an
# arbitrary Cypher string) are invoked INLINE without a CALL, so the keyword
# list above misses them. Reject the whole namespace by prefix. `db.` is NOT
# blocked - it's a common variable name (`MATCH (db:Database) RETURN db.x`)
# and its procedures need CALL (already blocked).
_FORBIDDEN_NAMESPACE_PATTERN = re.compile(r"\b(?:apoc|dbms)\s*\.", re.IGNORECASE)

# A single- or double-quoted Cypher string literal, honoring backslash escapes
# so an escaped quote inside the string (e.g. 'it\'s') does not end it early.
# Used to blank literal CONTENT before the forbidden-keyword / namespace scan.
# This is safe: a real write clause or procedure call can never live inside a
# string literal (there it is data, never executed), so blanking literals
# removes only false positives, never a real threat. An UNTERMINATED quote does
# not match, so nothing is stripped and the raw text is scanned as-is — a write
# keyword after a dangling quote still rejects (fail-closed).
_STRING_LITERAL_PATTERN = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")


def _strip_string_literals(cypher: str) -> str:
    """Replace each string literal's content with an empty quote pair so the
    forbidden-keyword / namespace scans ignore keywords that appear only as
    data (an entity named 'gene set', a value like 'apoc.tld', etc.)."""
    return _STRING_LITERAL_PATTERN.sub("''", cypher)


def is_cypher_read_only(cypher: str) -> bool:
    """Return True iff the Cypher contains no forbidden token.

    Rejects on any write keyword, `CALL`/`LOAD`, or the `apoc.`/`dbms.`
    namespace. String literals are blanked first, so a keyword that appears
    only as data (`WHERE n.name = "gene set"`) does NOT reject, while a real
    write clause outside any string still does. Word-boundary match: a property
    name like `created_at` does NOT trigger a rejection (no boundary between
    `created` and `_at`), but `CREATE`, `Create`, `create`,
    `CALL apoc.load.json(...)`, `LOAD CSV`, and inline `apoc.cypher.*(...)` all
    do.
    """
    scannable = _strip_string_literals(cypher)
    return (
        _FORBIDDEN_PATTERN.search(scannable) is None
        and _FORBIDDEN_NAMESPACE_PATTERN.search(scannable) is None
    )


def wrap_with_limit(cypher: str, limit: int) -> str:
    """Wrap a read Cypher query in a CALL subquery with a final LIMIT.

    The outer LIMIT is enforced by Neo4j itself, so this caps the rows
    returned regardless of whether the inner query has its own LIMIT
    clause. The caller is responsible for passing a sensible `limit`
    (typically `settings.kg_max_rows`).
    """
    return f"CALL {{\n{cypher}\n}}\nRETURN * LIMIT {limit}"
