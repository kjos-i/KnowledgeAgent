"""Tests for `kg/cypher_safety.py` - keyword validator + LIMIT wrapper.

Two test styles in this file:

  - Example-based tests (the original suite) — pin specific known
    cases: each forbidden keyword in turn, case variants, the
    `created_at` property that shares letters with CREATE.
  - Property-based tests at the bottom (via `hypothesis`) — generate
    inputs and assert invariants (any text with no forbidden token
    passes; any text with one fails; wrap_with_limit always wraps).

Property-based tests catch boundary cases the hand-picked examples
miss (e.g., keywords as substring of property names with non-word
boundary chars, mixed-case keywords inside strings, etc.).
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from knowledge_agent.kg.cypher_safety import (
    FORBIDDEN_KEYWORDS,
    is_cypher_read_only,
    wrap_with_limit,
)

# ---- is_cypher_read_only ----


def test_read_only_query_passes():
    assert is_cypher_read_only("MATCH (d:Document) RETURN d.doc_id LIMIT 10")


def test_empty_string_passes():
    assert is_cypher_read_only("")


def test_each_forbidden_keyword_rejects():
    """Every keyword in FORBIDDEN_KEYWORDS rejects Cypher containing it."""
    for kw in FORBIDDEN_KEYWORDS:
        cypher = f"MATCH (n) {kw} n.x = 1 RETURN n"
        assert not is_cypher_read_only(cypher), f"Forbidden keyword {kw!r} should reject the cypher"


def test_keyword_rejection_is_case_insensitive():
    for variant in ("CREATE", "create", "Create", "CrEaTe"):
        cypher = f"{variant} (n:X) RETURN n"
        assert not is_cypher_read_only(cypher), f"Case variant {variant!r} should reject"


def test_property_named_after_keyword_does_not_reject():
    """A property named `created_at` shares letters with CREATE but the
    word-boundary regex protects it."""
    cypher = "MATCH (d:Document) RETURN d.created_at LIMIT 10"
    assert is_cypher_read_only(cypher)


def test_keyword_at_start_of_query_rejects():
    assert not is_cypher_read_only("DELETE (n)")


def test_keyword_in_middle_of_query_rejects():
    cypher = "MATCH (n) WITH n SET n.x = 1 RETURN n"
    assert not is_cypher_read_only(cypher)


# ---- is_cypher_read_only: keywords inside string literals are data ----


def test_write_keyword_inside_double_quoted_literal_does_not_reject():
    """A forbidden keyword that appears only as string DATA (an entity named
    "gene set" — SET is not a write clause here) must NOT reject the read.
    Regression: the raw regex matched inside literals, so every such legit read
    was dropped and the KG leg returned empty."""
    assert is_cypher_read_only('MATCH (n:Entity) WHERE n.name = "gene set" RETURN n')


def test_write_keyword_inside_single_quoted_literal_does_not_reject():
    assert is_cypher_read_only("MATCH (n:Entity) WHERE n.name = 'gene set' RETURN n LIMIT 5")


def test_apoc_namespace_inside_string_literal_does_not_reject():
    """`apoc.` as a substring of a string VALUE is data, not an inline call, so
    the namespace check must not reject it once literals are blanked."""
    assert is_cypher_read_only("MATCH (n) WHERE n.host = 'apoc.example.com' RETURN n")


def test_real_write_after_string_literal_still_rejects():
    """Blanking literals must NOT weaken the guard: a real SET clause OUTSIDE
    the string is still caught."""
    assert not is_cypher_read_only("MATCH (n) WHERE n.x = 'ok' SET n.y = 1 RETURN n")


def test_unterminated_string_literal_fails_closed():
    """A dangling quote does not match the literal pattern, so nothing is
    stripped and the raw text is scanned — a write keyword after it still
    rejects (fail-closed, never fail-open)."""
    assert not is_cypher_read_only('MATCH (n) WHERE n.x = "abc DELETE (m) RETURN n')


def test_real_apoc_call_with_string_arg_still_rejects():
    """The dangerous `apoc.`/`CALL` tokens sit OUTSIDE the string argument, so
    blanking the URL literal leaves them to be caught."""
    assert not is_cypher_read_only("CALL apoc.load.json('http://169.254.169.254/latest/') YIELD v")


# ---- is_cypher_read_only: procedure / bulk-load / namespace hardening ----


def test_call_procedure_rejected():
    """CALL reaches any procedure (apoc.load.* = SSRF/file-read, dbms.*), so
    it's rejected even though it's not a write keyword."""
    cypher = "CALL apoc.load.json('http://169.254.169.254/latest/') YIELD value RETURN value"
    assert not is_cypher_read_only(cypher)


def test_load_csv_rejected():
    """LOAD CSV pulls a remote/local file into the read transaction."""
    assert not is_cypher_read_only("LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row")


def test_apoc_inline_function_rejected():
    """apoc.cypher.* is an INLINE function (no CALL) that runs an arbitrary
    Cypher string — the namespace check catches it where the keyword list
    can't. (Embedded read here so the rejection is driven ONLY by the `apoc.`
    namespace, not a keyword leaking from the payload.)"""
    cypher = "RETURN apoc.cypher.runFirstColumn('MATCH (n) RETURN n LIMIT 1', {}) AS x"
    assert not is_cypher_read_only(cypher)


def test_dbms_namespace_rejected():
    assert not is_cypher_read_only("CALL dbms.components() YIELD name RETURN name")
    assert not is_cypher_read_only("RETURN dbms.database.state('neo4j') AS s")


def test_call_and_load_are_case_insensitive():
    assert not is_cypher_read_only("caLL apoc.load.jdbc('jdbc:...')")
    assert not is_cypher_read_only("load csv from 'x' as r return r")


def test_db_variable_is_not_a_false_positive():
    """`db` is a common node/var name; only `apoc.`/`dbms.` are namespace-
    blocked and `db.*` procedures need CALL (already blocked), so a plain
    `MATCH (db:Database) RETURN db.name` read still passes."""
    assert is_cypher_read_only("MATCH (db:Database) RETURN db.name LIMIT 10")


@pytest.mark.security
def test_show_metadata_commands_rejected():
    """SHOW USERS / DATABASES / INDEXES / CONSTRAINTS leak schema and security
    metadata; the guard must block them (audit finding 1)."""
    for cmd in ("SHOW USERS", "SHOW DATABASES", "SHOW INDEXES", "show constraints"):
        assert not is_cypher_read_only(f"{cmd} YIELD * RETURN *"), f"{cmd!r} should reject"


# ---- wrap_with_limit ----


def test_wrap_produces_call_subquery_structure():
    out = wrap_with_limit("MATCH (d) RETURN d", 50)
    assert out.startswith("CALL {")
    assert "MATCH (d) RETURN d" in out
    assert out.endswith("RETURN * LIMIT 50")


def test_wrap_uses_provided_limit():
    assert "LIMIT 7" in wrap_with_limit("RETURN 1", 7)
    assert "LIMIT 1000" in wrap_with_limit("RETURN 1", 1000)


def test_wrap_preserves_multiline_inner_cypher():
    inner = "MATCH (a:Author)\nWHERE a.name = 'X'\nRETURN a"
    out = wrap_with_limit(inner, 50)
    assert inner in out


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis).
#
# Strategy: build inputs from a "safe alphabet" that excludes
# every forbidden keyword (and case variants). For "must-reject"
# tests, inject ONE keyword into otherwise-safe text and assert
# rejection. For "must-accept" tests, generate pure safe text.
# ---------------------------------------------------------------------------

# Safe alphabet: ASCII letters that don't form any forbidden keyword
# when concatenated AS substrings. Restricting to lowercase letters
# x-z + digits keeps strategies from accidentally building "create"
# or similar.
_SAFE_CHARS = "xyz0123456789_ \n()[].:,"
_SAFE_TEXT = st.text(alphabet=_SAFE_CHARS, min_size=0, max_size=100)


def _case_permutations(word: str) -> st.SearchStrategy[str]:
    """Strategy: yield every case-permutation of `word`.

    For `set` (3 chars), 8 variants: set, Set, sEt, ..., SET. Used to
    pin that the validator's case-insensitivity holds for arbitrary
    case mixings of every forbidden keyword.
    """
    return st.tuples(*[st.sampled_from([c.lower(), c.upper()]) for c in word]).map(
        lambda parts: "".join(parts)
    )


@given(safe=_SAFE_TEXT)
def test_safe_text_always_passes_validator(safe: str) -> None:
    """Text built from the safe alphabet (no forbidden keyword can
    appear as a token) always passes the read-only check."""
    assert is_cypher_read_only(safe)


@given(
    keyword=st.sampled_from(FORBIDDEN_KEYWORDS),
    prefix=_SAFE_TEXT,
    suffix=_SAFE_TEXT,
)
def test_text_with_forbidden_keyword_always_rejects(keyword: str, prefix: str, suffix: str) -> None:
    """Any text that embeds a forbidden keyword at a word boundary
    (between safe-alphabet characters that include whitespace + line
    breaks) must be rejected — regardless of where in the string the
    keyword appears."""
    cypher = f"{prefix} {keyword} {suffix}"
    assert not is_cypher_read_only(cypher)


@given(
    keyword=st.sampled_from(FORBIDDEN_KEYWORDS),
    prefix=_SAFE_TEXT,
    suffix=_SAFE_TEXT,
)
def test_case_variants_of_forbidden_keyword_reject(keyword: str, prefix: str, suffix: str) -> None:
    """For each case permutation of a forbidden keyword (e.g. cReAtE),
    the validator still rejects — locks case-insensitive detection."""
    permuted = "".join(c.upper() if (i % 2 == 0) else c.lower() for i, c in enumerate(keyword))
    cypher = f"{prefix} {permuted} {suffix}"
    assert not is_cypher_read_only(cypher)


@given(
    inner=st.text(alphabet=_SAFE_CHARS, min_size=1, max_size=200),
    limit=st.integers(min_value=1, max_value=10_000),
)
def test_wrap_with_limit_preserves_inner_and_caps_outer(inner: str, limit: int) -> None:
    """For any safe inner Cypher + any positive limit, the wrapped
    output contains the inner string verbatim AND ends with the
    LIMIT N clause carrying the requested N."""
    out = wrap_with_limit(inner, limit)
    assert inner in out
    assert out.endswith(f"RETURN * LIMIT {limit}")
    assert out.startswith("CALL {")
