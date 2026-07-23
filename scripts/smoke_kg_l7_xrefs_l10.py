"""Smoke test for L7 cross-ontology xrefs + L10 cross_doc_xrefs (with
manual-inspection pause).

End-to-end exercise of the xrefs / L10 ship:

  L7  - Force re-imports a small ontology subset (HPO + MONDO + MeSH)
        with `xrefs_mode="use"`. MONDO is a particularly rich xref
        source — it declares equivalents in MeSH / HPO / DOID / OMIM
        as core graph data — so the resolved-edge count should be
        well above zero for any subset that includes it.

  L7+ - Runs `backfill_resolved_xrefs(client)` to verify the
        idempotency contract: the first call (right after import)
        should still find some work (xrefs whose targets weren't
        present at import time but ARE now because we imported all
        three ontologies in this run), the second call should find
        nothing.

  L7+ - Demonstrates `clear_xref_edges_for_ontology(client, "MeSHTerm")`
        and verifies the count drops to zero for MeSH's outgoing
        xref edges.

  L10 - Optionally runs `recompute_cross_doc_xrefs_global(client,
        threshold)` to check the Cypher executes cleanly. WITHOUT
        ingested documents the result is always 0 edges — that's a
        valid smoke outcome (no docs to relate). End-to-end L10 with
        real docs is covered by `smoke_pipeline.py` after this ship.

WARNING: heavy. First run downloads HPO (~40 MB) + MONDO (~30 MB) +
MeSH (~115 MB) and parses them. Subsequent runs reuse the cache so
the smoke completes in a minute or two. Total xref strings extracted
across these three ontologies: ~hundreds of thousands. Memory: a few
GB while parsing MeSH N-Triples.

Requires Neo4j running and NEO4J_PASSWORD set in `.env.test` per
[[test-instance-setup]]. The script switches to the test instance
BEFORE any other knowledge_agent import that might trigger `get_settings()`.

Lifecycle:
  1. Clear any leftover ontology terms from the 3 ontologies.
  2. Force re-import with `xrefs_mode="use"` (writes :<X>_XREF edges).
  3. Run backfill + idempotency assertions.
  4. Demo clear_xref_edges_for_ontology on MeSH.
  5. Optional: recompute L10 graph-wide (--include-l10).
  6. Pause - you inspect in Neo4j Desktop.
  7. Press Enter to clean up, Ctrl+C to keep the nodes for further poking.

Run from the project root:
    python scripts/smoke_kg_l7_xrefs_l10.py
    python scripts/smoke_kg_l7_xrefs_l10.py --include-l10
    python scripts/smoke_kg_l7_xrefs_l10.py --keep-cache    # don't force re-download

Automated counterparts (for regression catching, no real ontology
imports — use synthetic OntologyTerm + canonical entities):
  tests/integration/kg/test_xrefs.py             (backfill / clear /
                                                  count primitives;
                                                  dangling-strip
                                                  idempotency)
  tests/integration/kg/test_cross_doc_xrefs.py   (L10 recompute via
                                                  identity OR xref-edge
                                                  equivalence; threshold)
The smoke remains the only place that exercises the heavy real
import paths end-to-end.
Run via `pytest -m integration tests/integration/kg/`.
"""

import argparse
import asyncio
import sys

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# knowledge_agent import that might trigger `get_settings()`. Per
# [[test-instance-setup]] the test instance has a different password
# so a wrong-instance state fails auth rather than corrupting real
# data.
from knowledge_agent.config import load_test_env

load_test_env()

# Initialise the logging system: rotating file at the OS-standard log
# dir (override with KAGENT_LOG_DIR=./logs to land them inside the repo),
# crash files in <log_dir>/crashes/, library noise clamped, ring
# buffer at DEBUG for crash context. This smoke is sync so we don't
# wire the returned asyncio handler — that's the Flet entry point's
# job. See `logging_setup.init_logging` docstring for the async
# wiring pattern.
from knowledge_agent.logging_setup import init_logging  # noqa: E402

init_logging()

from knowledge_agent.kg import cross_doc_xrefs_writes  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.ontologies import (  # noqa: E402
    hpo_writes,
    mesh_writes,
    mondo_writes,
    xrefs,
)
from knowledge_agent.kg.schema import (  # noqa: E402
    HPO_TERM_LABEL,
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    ONTOLOGY_XREF_RELS,
)


def _count_all_xref_edges(client) -> int:
    """Live count of every `:<X>_XREF` edge across all 18 types via
    the pipe-union diagnostic from `kg.ontologies.xrefs`."""
    n = xrefs.count_xref_edges(client, None)
    return n if n is not None else 0


def _count_one_xref_type(client, term_label: str) -> int:
    n = xrefs.count_xref_edges(client, term_label)
    return n if n is not None else 0


def _count_dangling(client, term_label: str) -> int:
    n = xrefs.count_dangling_xrefs(client, term_label)
    return n if n is not None else 0


def _clear_terms(client) -> None:
    """Wipe the three ontologies we're about to (re)import. Idempotent
    — DETACH DELETE on empty match-set is a no-op."""
    for delete_fn, name in (
        (hpo_writes.delete_imported, "HPO"),
        (mondo_writes.delete_imported, "MONDO"),
        (mesh_writes.delete_imported, "MeSH"),
    ):
        ok = delete_fn(client)
        print(f"  clear {name} -> {ok}")


def _summary(client) -> None:
    print("  Current xref-edge counts:")
    total = _count_all_xref_edges(client)
    print(f"    total (all 18 :<X>_XREF types) = {total}")
    for label in (HPO_TERM_LABEL, MONDO_TERM_LABEL, MESH_TERM_LABEL):
        n = _count_one_xref_type(client, label)
        dangling = _count_dangling(client, label)
        print(
            f"    {label:>14}: {n} outgoing edges, "
            f"{dangling} sources with dangling strings remaining"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-l10",
        action="store_true",
        help=(
            "Also run `recompute_cross_doc_xrefs_global` (L10). "
            "Without ingested documents the result is 0 edges — the "
            "smoke just verifies the Cypher executes cleanly."
        ),
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help=(
            "Don't force re-download of the source files. Default is "
            "to reuse anything in the local cache; this flag is a "
            "no-op today (kept for symmetry with future --force-cache)."
        ),
    )
    parser.add_argument(
        "--l10-threshold",
        type=int,
        default=2,
        help="Threshold for the optional L10 recompute (default 2).",
    )
    args = parser.parse_args()
    _ = args.keep_cache  # currently informational only

    client = get_kg_client()

    print("Applying constraints...")
    await client.ensure_constraints()
    print("  ensure_constraints -> ok")

    print()
    print("Clearing prior HPO + MONDO + MeSH term nodes (if any)...")
    _clear_terms(client)

    print()
    print("Importing HPO with xrefs_mode='use'...")
    try:
        hpo_writes.import_hpo(
            client,
            force=True,
            xrefs_mode="use",
        )
    except Exception as exc:
        print(f"  HPO import failed: {exc!r} — abort.", file=sys.stderr)
        await client.close()
        sys.exit(1)
    print()
    print("Importing MONDO with xrefs_mode='use'...")
    try:
        mondo_writes.import_mondo(
            client,
            force=True,
            xrefs_mode="use",
        )
    except Exception as exc:
        print(f"  MONDO import failed: {exc!r} — abort.", file=sys.stderr)
        await client.close()
        sys.exit(1)
    print()
    print("Importing MeSH with xrefs_mode='use'...")
    try:
        await mesh_writes.import_mesh(
            client,
            force=True,
            xrefs_mode="use",
        )
    except Exception as exc:
        print(f"  MeSH import failed: {exc!r} — abort.", file=sys.stderr)
        await client.close()
        sys.exit(1)

    print()
    print("Post-import state:")
    _summary(client)

    print()
    print("Running backfill_resolved_xrefs (call #1)...")
    first = xrefs.backfill_resolved_xrefs(client)
    if first is None:
        print("  backfill returned None — session error.", file=sys.stderr)
        await client.close()
        sys.exit(1)
    n1_attempted = sum(r["n_edges_attempted"] for r in first.values())
    n1_cleaned = sum(r["n_sources_cleaned"] for r in first.values())
    print(
        f"  edges attempted across all ontologies: {n1_attempted}; "
        f"source nodes cleaned: {n1_cleaned}"
    )

    print()
    print("Running backfill_resolved_xrefs (call #2 — idempotency check)...")
    second = xrefs.backfill_resolved_xrefs(client)
    if second is None:
        print("  backfill returned None on call #2.", file=sys.stderr)
        await client.close()
        sys.exit(1)
    n2_cleaned = sum(r["n_sources_cleaned"] for r in second.values())
    if n2_cleaned == 0:
        print(
            f"  edges attempted: "
            f"{sum(r['n_edges_attempted'] for r in second.values())} "
            "(MERGE-idempotent, expected), sources cleaned: 0 (idempotent ✓)"
        )
    else:
        print(
            f"  WARNING: second backfill cleaned {n2_cleaned} more sources. "
            "Idempotency contract suspect — investigate."
        )

    print()
    print("Demonstrating clear_xref_edges_for_ontology(MeSHTerm)...")
    pre_mesh = _count_one_xref_type(client, MESH_TERM_LABEL)
    print(f"  before clear: {pre_mesh} outgoing :MESH_XREF edges")
    n_cleared = await xrefs.clear_xref_edges_for_ontology(
        client,
        MESH_TERM_LABEL,
    )
    post_mesh = _count_one_xref_type(client, MESH_TERM_LABEL)
    print(
        f"  clear_xref_edges_for_ontology returned: {n_cleared}; "
        f"after clear: {post_mesh} outgoing :MESH_XREF edges"
    )
    if post_mesh != 0:
        print(
            "  WARNING: MeSH outgoing xref edges did not drop to 0 after clear. Investigate.",
            file=sys.stderr,
        )

    if args.include_l10:
        print()
        print(f"Running recompute_cross_doc_xrefs_global (L10, threshold={args.l10_threshold})...")
        n_l10 = cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(
            client,
            args.l10_threshold,
        )
        if n_l10 is None:
            print("  L10 recompute returned None — session error.", file=sys.stderr)
        else:
            print(
                f"  :RELATED_BY_XREF edges written: {n_l10}. "
                "(Expected 0 unless ingested docs sharing canonical "
                "concepts are present.)"
            )

    print()
    print("Smoke data written. In Neo4j Desktop -> Query, try:")
    print("  // Top 10 MONDO terms with the most outgoing :MONDO_XREF edges:")
    print("  MATCH (s:MONDOTerm)-[r:MONDO_XREF]->(t:OntologyTerm)")
    print("  RETURN s.id, s.label, count(r) AS n ORDER BY n DESC LIMIT 10")
    print()
    print("  // Cross-source xref traversal via pipe-union (Neo4j 5+):")
    print(
        "  MATCH (a:OntologyTerm)-[r:"
        + "|".join(ONTOLOGY_XREF_RELS)
        + "]-(b:OntologyTerm) RETURN type(r), count(r) AS n ORDER BY n DESC"
    )
    print()
    print("  // Surface MONDO terms whose dangling_xrefs still hold unresolved strings:")
    print("  MATCH (s:MONDOTerm) WHERE size(s.dangling_xrefs) > 0")
    print("  RETURN s.id, s.dangling_xrefs LIMIT 20")
    print()

    try:
        input("Press Enter to delete the smoke ontology data, Ctrl+C to keep it. ")
    except KeyboardInterrupt:
        print()
        print(
            "Keeping smoke data. Re-run this script (which clears at start) to clean it up later."
        )
        await client.close()
        return

    print("Deleting smoke ontology data...")
    _clear_terms(client)
    print("Done.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
