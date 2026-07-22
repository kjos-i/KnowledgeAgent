"""Integration tests for real ontology imports — one per format path.

The 18-ontology menu uses four distinct format paths:

  - **pronto OBO** — 13 ontologies (HPO, MONDO, ChEBI, GO, UBERON, ECO,
    SO, PR, CL, PO, FOODON, ENVO, NCBITaxon). Each downloads a single
    `.obo` file and parses it via pronto.
  - **rdflib OWL** — 3 ontologies (OBI, EFO, DRON). Each downloads an
    OWL/RDF-XML file and parses it via rdflib (pronto drops oboInOwl
    synonyms in OWL, so this path was built separately).
  - **MeSH RDF (streaming N-Triples)** — 1 ontology (MeSH). Uses a
    custom streaming N-Triples sink to handle the ~600 MB download
    without loading the whole graph into memory.
  - **FIBO multi-file walker** — 1 ontology (FIBO). Walks the FIBO
    GitHub repo, fetches ~70 `.rdf` files, loads them into one rdflib
    Graph, then extracts terms.

We exercise each path against the SMALLEST representative ontology
to cap total run time. Each test marked `@pytest.mark.slow` —
opt in via `pytest -m "integration and slow"`. First run downloads
the ontology to the `ontology_downloads_dir` default (the OS cache
dir); subsequent runs reuse it.

Manual interactive counterpart: `scripts/smoke_kg_l7_xrefs_l10.py`
exercises HPO + MONDO + MeSH together with full xref backfill +
L10 recompute.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.kg import (
    ontology_efo_writes,
    ontology_fibo_writes,
    ontology_hpo_writes,
    ontology_mesh_writes,
)
from knowledge_agent.kg.schema import (
    EFO_TERM_LABEL,
    FIBO_TERM_LABEL,
    HPO_TERM_LABEL,
    MESH_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _skip_unless_downloaded(cache_name: str, ontology_name: str) -> None:
    """Skip when the ontology's source file/dir isn't in the downloads dir.

    These `slow` tests import a REAL ontology and never auto-download
    (`require_cached` is the read-only, never-fetch probe). In a bare
    environment (fresh clone / CI) the file is absent — skip with the same
    actionable message ingest surfaces, rather than hard-fail. On a machine
    where the ontology was downloaded via Library → Installs, the test runs."""
    from knowledge_agent.kg.ontology_helpers import (
        OntologyNotDownloadedError,
        require_cached,
    )

    try:
        require_cached(cache_name, ontology_name)
    except OntologyNotDownloadedError as exc:
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------
# pronto OBO path: HPO (~30 MB, smallest of the 13 pronto ontologies).
# ---------------------------------------------------------------------------


async def test_import_hpo_writes_term_nodes_via_pronto_obo_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_hpo downloads + parses the OBO + writes :HPOTerm nodes.
    Asserts > 0 terms imported and the is_imported helper agrees."""
    _skip_unless_downloaded(ontology_hpo_writes.HPO_CACHE_FILENAME, "HPO")

    # Clean: drop any prior HPO state so the test exercises the
    # write path (not the already-imported short-circuit).
    await ontology_hpo_writes.delete_imported(kg_client)

    ok = await ontology_hpo_writes.import_hpo(kg_client, force=True)
    assert ok is True

    # is_imported now returns True.
    assert await ontology_hpo_writes.is_imported(kg_client) is True

    # Count HPO term nodes — should be in the thousands.
    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n")
        n_terms = (await result.single())["n"]
    assert n_terms > 1000  # HPO has ~16K terms

    # All HPO terms carry the OntologyTerm super-label too.
    async with kg_client.driver.session() as session:
        result = await session.run(
            f"MATCH (t:{ONTOLOGY_TERM_LABEL}) WHERE t.id STARTS WITH 'HP:' RETURN count(t) AS n"
        )
        n_super = (await result.single())["n"]
    assert n_super == n_terms


async def test_hpo_import_idempotent_via_ensure_imported(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """Calling import_hpo a second time without force=True should
    short-circuit via is_imported and not duplicate any nodes."""
    _skip_unless_downloaded(ontology_hpo_writes.HPO_CACHE_FILENAME, "HPO")
    await ontology_hpo_writes.delete_imported(kg_client)
    await ontology_hpo_writes.import_hpo(kg_client, force=True)

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n")
        n_before = (await result.single())["n"]

    # Second call without force; should not write anything.
    await ontology_hpo_writes.import_hpo(kg_client, force=False)

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n")
        n_after = (await result.single())["n"]
    assert n_after == n_before


async def test_hpo_delete_imported_drops_every_term_node(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """delete_imported runs the full :HPOTerm wipe + relationship
    cleanup. After call, is_imported returns False."""
    _skip_unless_downloaded(ontology_hpo_writes.HPO_CACHE_FILENAME, "HPO")
    await ontology_hpo_writes.import_hpo(kg_client, force=True)
    assert await ontology_hpo_writes.is_imported(kg_client) is True

    await ontology_hpo_writes.delete_imported(kg_client)
    assert await ontology_hpo_writes.is_imported(kg_client) is False

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n")
        n = (await result.single())["n"]
    assert n == 0


# ---------------------------------------------------------------------------
# rdflib OWL path: EFO (~50 MB, smallest of the 3 rdflib ontologies).
# ---------------------------------------------------------------------------


async def test_import_efo_writes_term_nodes_via_rdflib_owl_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_efo exercises the rdflib OWL path (pronto drops
    oboInOwl synonyms on OWL files, so we built `extract_terms_owl`
    + `read_rdf` separately for the 3 OWL ontologies). Asserts > 0
    terms imported."""
    _skip_unless_downloaded(ontology_efo_writes.EFO_CACHE_FILENAME, "EFO")
    await ontology_efo_writes.delete_imported(kg_client)
    ok = await ontology_efo_writes.import_efo(kg_client, force=True)
    assert ok is True

    assert await ontology_efo_writes.is_imported(kg_client) is True

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{EFO_TERM_LABEL}) RETURN count(t) AS n")
        n = (await result.single())["n"]
    assert n > 1000  # EFO has tens of thousands of terms


# ---------------------------------------------------------------------------
# MeSH RDF streaming N-Triples path.
# ---------------------------------------------------------------------------


async def test_import_mesh_writes_term_nodes_via_streaming_n_triples_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_mesh exercises the custom streaming N-Triples sink
    (~600 MB download). Asserts > 0 :MeSHTerm nodes after import."""
    _skip_unless_downloaded(ontology_mesh_writes.MESH_CACHE_FILENAME, "MeSH")
    await ontology_mesh_writes.delete_imported(kg_client)
    ok = await ontology_mesh_writes.import_mesh(kg_client, force=True)
    assert ok is True

    assert await ontology_mesh_writes.is_imported(kg_client) is True

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{MESH_TERM_LABEL}) RETURN count(t) AS n")
        n = (await result.single())["n"]
    assert n > 20000  # MeSH has ~30K Descriptors


# ---------------------------------------------------------------------------
# FIBO multi-file GitHub walker path.
# ---------------------------------------------------------------------------


async def test_import_fibo_writes_term_nodes_via_multi_file_walker_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_fibo exercises the GitHub walker that fetches ~70
    `.rdf` files and loads them into one rdflib Graph. Asserts > 0
    :FIBOTerm nodes after import."""
    _skip_unless_downloaded(ontology_fibo_writes.FIBO_CACHE_SUBDIR, "FIBO")
    await ontology_fibo_writes.delete_imported(kg_client)
    ok = await ontology_fibo_writes.import_fibo(kg_client, force=True)
    assert ok is True

    assert await ontology_fibo_writes.is_imported(kg_client) is True

    async with kg_client.driver.session() as session:
        result = await session.run(f"MATCH (t:{FIBO_TERM_LABEL}) RETURN count(t) AS n")
        n = (await result.single())["n"]
    assert n > 100  # FIBO has thousands of classes across modules
