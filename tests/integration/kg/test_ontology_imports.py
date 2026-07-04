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
the ontology to `~/.research-literature-agent/ontology-downloads/`
(the ontology_downloads_dir default); subsequent runs reuse it.

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


# ---------------------------------------------------------------------------
# pronto OBO path: HPO (~30 MB, smallest of the 13 pronto ontologies).
# ---------------------------------------------------------------------------


def test_import_hpo_writes_term_nodes_via_pronto_obo_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_hpo downloads + parses the OBO + writes :HPOTerm nodes.
    Asserts > 0 terms imported and the is_imported helper agrees."""
    # Clean: drop any prior HPO state so the test exercises the
    # write path (not the already-imported short-circuit).
    ontology_hpo_writes.delete_imported(kg_client)

    ok = ontology_hpo_writes.import_hpo(kg_client, force=True)
    assert ok is True

    # is_imported now returns True.
    assert ontology_hpo_writes.is_imported(kg_client) is True

    # Count HPO term nodes — should be in the thousands.
    with kg_client.driver.session() as session:
        n_terms = session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
    assert n_terms > 1000  # HPO has ~16K terms

    # All HPO terms carry the OntologyTerm super-label too.
    with kg_client.driver.session() as session:
        n_super = session.run(
            f"MATCH (t:{ONTOLOGY_TERM_LABEL}) WHERE t.id STARTS WITH 'HP:' RETURN count(t) AS n"
        ).single()["n"]
    assert n_super == n_terms


def test_hpo_import_idempotent_via_ensure_imported(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """Calling import_hpo a second time without force=True should
    short-circuit via is_imported and not duplicate any nodes."""
    ontology_hpo_writes.delete_imported(kg_client)
    ontology_hpo_writes.import_hpo(kg_client, force=True)

    with kg_client.driver.session() as session:
        n_before = session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]

    # Second call without force; should not write anything.
    ontology_hpo_writes.import_hpo(kg_client, force=False)

    with kg_client.driver.session() as session:
        n_after = session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
    assert n_after == n_before


def test_hpo_delete_imported_drops_every_term_node(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """delete_imported runs the full :HPOTerm wipe + relationship
    cleanup. After call, is_imported returns False."""
    ontology_hpo_writes.import_hpo(kg_client, force=True)
    assert ontology_hpo_writes.is_imported(kg_client) is True

    ontology_hpo_writes.delete_imported(kg_client)
    assert ontology_hpo_writes.is_imported(kg_client) is False

    with kg_client.driver.session() as session:
        n = session.run(f"MATCH (t:{HPO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
    assert n == 0


# ---------------------------------------------------------------------------
# rdflib OWL path: EFO (~50 MB, smallest of the 3 rdflib ontologies).
# ---------------------------------------------------------------------------


def test_import_efo_writes_term_nodes_via_rdflib_owl_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_efo exercises the rdflib OWL path (pronto drops
    oboInOwl synonyms on OWL files, so we built `extract_terms_owl`
    + `read_rdf` separately for the 3 OWL ontologies). Asserts > 0
    terms imported."""
    ontology_efo_writes.delete_imported(kg_client)
    ok = ontology_efo_writes.import_efo(kg_client, force=True)
    assert ok is True

    assert ontology_efo_writes.is_imported(kg_client) is True

    with kg_client.driver.session() as session:
        n = session.run(f"MATCH (t:{EFO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
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
    ontology_mesh_writes.delete_imported(kg_client)
    ok = await ontology_mesh_writes.import_mesh(kg_client, force=True)
    assert ok is True

    assert ontology_mesh_writes.is_imported(kg_client) is True

    with kg_client.driver.session() as session:
        n = session.run(f"MATCH (t:{MESH_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
    assert n > 20000  # MeSH has ~30K Descriptors


# ---------------------------------------------------------------------------
# FIBO multi-file GitHub walker path.
# ---------------------------------------------------------------------------


def test_import_fibo_writes_term_nodes_via_multi_file_walker_path(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """import_fibo exercises the GitHub walker that fetches ~70
    `.rdf` files and loads them into one rdflib Graph. Asserts > 0
    :FIBOTerm nodes after import."""
    ontology_fibo_writes.delete_imported(kg_client)
    ok = ontology_fibo_writes.import_fibo(kg_client, force=True)
    assert ok is True

    assert ontology_fibo_writes.is_imported(kg_client) is True

    with kg_client.driver.session() as session:
        n = session.run(f"MATCH (t:{FIBO_TERM_LABEL}) RETURN count(t) AS n").single()["n"]
    assert n > 100  # FIBO has thousands of classes across modules
