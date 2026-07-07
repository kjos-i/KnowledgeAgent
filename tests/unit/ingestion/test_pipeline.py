"""Tests for ingestion.pipeline - helpers + IngestResult dataclass + gating.

End-to-end behaviour of `ingest_document` (real LanceDB / Neo4j / OpenAlex /
Voyage / docling) is exercised by `scripts/smoke_pipeline.py`. The gating
tests at the bottom of this file are narrower: they patch all the heavy
deps and verify only the config-driven dispatch logic - which KG writes
are called vs skipped for a given `CorpusConfig`.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.ingestion.parse import ParsedChunk
from knowledge_agent.ingestion.pipeline import (
    IngestResult,
    _build_authors_display,
    _build_lance_rows,
    _doc_metadata_fields_from_work,
    backfill_chunks,
    backfill_cross_doc,
    backfill_cross_doc_xrefs,
    backfill_entities,
    backfill_ontology,
    backfill_triples,
    delete_doc,
    ingest_document,
    lookup_known_doi,
    re_embed,
    resolve_openalex,
)
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
)
from knowledge_agent.kg.triples_writes import ExtractedTriple
from knowledge_agent.search.client import LanceClient


@pytest.fixture(autouse=True)
def _stub_search_client():
    """Safety net: no pipeline unit test may touch the real LanceDB.

    Several `ingest_document` tests mock parse / embed / KG / extractors
    but NOT the search client, so `pipeline.get_search_client()` fell
    through to a real `LanceClient` — which reads the dev `.env`
    (`LANCEDB_PATH=./lancedb`), did a `mkdir` + connect, and created a
    real `./lancedb` folder in the working directory. That violates the
    suite's invariant that unit tests never touch real data / the real
    `.env`. This autouse fixture stubs `pipeline.get_search_client` with
    an async-safe mock for every test in this module; tests that assert
    on the client still patch it explicitly and their patch wins in scope.
    """
    stub = MagicMock(spec=LanceClient)
    for _name in (
        "ensure_schema",
        "ensure_indexes",
        "write_chunks",
        "delete_chunks_by_doc_id",
        "update_doc_metadata",
        "drop_chunks_table",
        "close",
    ):
        setattr(stub, _name, AsyncMock(return_value=None))
    stub.get_chunks_by_doc_id = AsyncMock(return_value=[])
    stub.list_indexed_docs = AsyncMock(return_value=[])
    stub.retrieve = AsyncMock(return_value=[])
    with patch(
        "knowledge_agent.ingestion.pipeline.get_search_client",
        return_value=stub,
    ):
        yield


def _chunk(
    index: int, text: str, section: str | None = None, page: int | None = None
) -> ParsedChunk:
    return ParsedChunk(chunk_index=index, text=text, section=section, page=page)


# ---- _build_authors_display ----


def test_authors_display_single_author():
    auths = [{"author": {"display_name": "Alice"}}]
    assert _build_authors_display(auths) == "Alice"


def test_authors_display_three_authors_no_et_al():
    auths = [
        {"author": {"display_name": "Alice"}},
        {"author": {"display_name": "Bob"}},
        {"author": {"display_name": "Carol"}},
    ]
    assert _build_authors_display(auths) == "Alice, Bob, Carol"


def test_authors_display_four_or_more_appends_et_al():
    auths = [
        {"author": {"display_name": "Alice"}},
        {"author": {"display_name": "Bob"}},
        {"author": {"display_name": "Carol"}},
        {"author": {"display_name": "Dave"}},
    ]
    assert _build_authors_display(auths) == "Alice, Bob, Carol, et al."


def test_authors_display_empty_returns_none():
    assert _build_authors_display([]) is None


def test_authors_display_skips_authors_without_display_name():
    auths = [
        {"author": {}},
        {"author": {"display_name": "Alice"}},
    ]
    assert _build_authors_display(auths) == "Alice"


def test_authors_display_no_resolvable_names_returns_none():
    auths = [{"author": {}}, {"author": {}}]
    assert _build_authors_display(auths) is None


# ---- delete_doc ----


def _patch_delete_clients(search_ok: bool, kg_ok: dict[str, bool]):
    """Build patched search + kg client mocks for delete_doc tests.

    All 4 primitives are now migrated to the typed-errors contract:
    success = returns None; failure = raises. Pass `False` to inject
    a `RuntimeError("boom")` `side_effect` on the corresponding mock.
    `kg_ok` keys: "delete_doc", "delete_chunks", "delete_entities".
    """
    search_mock = MagicMock(spec=LanceClient)
    if search_ok:
        search_mock.delete_chunks_by_doc_id = AsyncMock(return_value=None)
    else:
        search_mock.delete_chunks_by_doc_id = AsyncMock(side_effect=RuntimeError("boom"))

    kg_mock = MagicMock(spec=Neo4jClient)
    for key, attr in (
        ("delete_doc", "delete_doc"),
        ("delete_chunks", "delete_chunks_by_doc_id"),
        ("delete_entities", "delete_entities_by_doc_id"),
    ):
        if kg_ok[key]:
            setattr(kg_mock, attr, AsyncMock(return_value=None))
        else:
            setattr(kg_mock, attr, AsyncMock(side_effect=RuntimeError("boom")))
    # delete_triples_by_doc_id is called as part of delete_doc's sequential gather;
    # keep it as a success-AsyncMock unless the test specifies otherwise.
    if not isinstance(kg_mock.delete_triples_by_doc_id, AsyncMock):
        kg_mock.delete_triples_by_doc_id = AsyncMock(return_value=None)

    return search_mock, kg_mock


async def test_delete_doc_calls_all_four_primitives():
    """delete_doc composes 4 store primitives - one LanceDB, three KG."""
    search_mock, kg_mock = _patch_delete_clients(
        True, {"delete_doc": True, "delete_chunks": True, "delete_entities": True}
    )
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await delete_doc("doc-xyz")

    assert result is True
    search_mock.delete_chunks_by_doc_id.assert_called_once_with("doc-xyz")
    kg_mock.delete_doc.assert_called_once_with("doc-xyz")
    kg_mock.delete_chunks_by_doc_id.assert_called_once_with("doc-xyz")
    kg_mock.delete_entities_by_doc_id.assert_called_once_with("doc-xyz")


async def test_delete_doc_returns_false_when_lancedb_delete_fails():
    """LanceDB primitive raising flows through the `_safe` wrapper
    and surfaces as overall False."""
    search_mock, kg_mock = _patch_delete_clients(
        False, {"delete_doc": True, "delete_chunks": True, "delete_entities": True}
    )
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await delete_doc("doc-xyz")

    assert result is False


async def test_delete_doc_runs_all_primitives_even_when_first_fails():
    """No short-circuit: a failed LanceDB delete must NOT skip the KG cleanup."""
    search_mock, kg_mock = _patch_delete_clients(
        False, {"delete_doc": True, "delete_chunks": True, "delete_entities": True}
    )
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        await delete_doc("doc-xyz")

    # Every primitive ran exactly once even though the first raised.
    search_mock.delete_chunks_by_doc_id.assert_called_once()
    kg_mock.delete_doc.assert_called_once()
    kg_mock.delete_chunks_by_doc_id.assert_called_once()
    kg_mock.delete_entities_by_doc_id.assert_called_once()


async def test_delete_doc_returns_false_when_any_kg_primitive_fails():
    """Any single primitive failure causes the overall return to be False.

    All migrated primitives (delete_doc, delete_chunks, delete_entities)
    signal failure by raising. The `delete_doc` pipeline orchestrator
    must convert each raise into `kg_step_ok=False` without aborting
    the other steps.
    """
    for failing_key in ("delete_doc", "delete_chunks", "delete_entities"):
        kg_returns = {
            "delete_doc": True,
            "delete_chunks": True,
            "delete_entities": True,
        }
        search_mock, kg_mock = _patch_delete_clients(True, kg_returns)
        if failing_key == "delete_doc":
            kg_mock.delete_doc = AsyncMock(side_effect=RuntimeError("boom"))
        elif failing_key == "delete_chunks":
            kg_mock.delete_chunks_by_doc_id = AsyncMock(side_effect=RuntimeError("boom"))
        elif failing_key == "delete_entities":
            kg_mock.delete_entities_by_doc_id = AsyncMock(side_effect=RuntimeError("boom"))
        with (
            patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
            patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        ):
            assert await delete_doc("doc-xyz") is False, (
                f"delete_doc should return False when {failing_key} fails"
            )


# ---- backfill_ontology ----


def _config_with_ontologies(*names: str) -> CorpusConfig:
    """Build a CorpusConfig with the given ontology layers enabled."""
    flags_kwargs = {"chunks": True, "entities": True}
    ontology_cfg = {}
    for n in names:
        flags_kwargs[f"ontology_{n}"] = True
        ontology_cfg[n] = OntologyConfig(matching="exact")
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(**flags_kwargs),
        entities=EntityConfig(extractor="llm"),
        ontology=ontology_cfg,
    )


async def test_backfill_ontology_iterates_each_enabled_ontology():
    """Both enabled ontologies get ensure_imported + link calls."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=(False, True))
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=7)

    config = _config_with_ontologies("mesh", "go")
    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        results = await backfill_ontology("doc-xyz", config)

    assert set(results.keys()) == {"mesh", "go"}
    assert kg_mock.ensure_ontology_imported.call_count == 2
    assert kg_mock.link_entities_to_ontology.call_count == 2


async def test_backfill_ontology_passes_doc_id_and_matching_strategy():
    """Link call must scope to doc_id and use the per-ontology matching."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=(False, True))
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=3)

    config = _config_with_ontologies("mesh")
    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        await backfill_ontology("doc-xyz", config)

    kg_mock.link_entities_to_ontology.assert_called_once_with("mesh", "exact", doc_id="doc-xyz")


async def test_backfill_ontology_result_shape_matches_full_ingest():
    """Per-ontology dict has the same keys as IngestResult.kg_ontology_results."""
    kg_mock = MagicMock(spec=Neo4jClient)
    # was_imported=False (just ran the import) -> "imported"=True in result.
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=False)
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=4)

    config = _config_with_ontologies("mesh")
    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        results = await backfill_ontology("doc-xyz", config)

    assert results == {
        "mesh": {"imported": True, "import_ok": True, "n_links": 4},
    }


async def test_backfill_ontology_one_ontology_failure_does_not_block_others():
    """If MeSH ensure_imported raises, GO still gets a link call."""
    kg_mock = MagicMock(spec=Neo4jClient)

    def ensure_side_effect(name, **kwargs):
        if name == "mesh":
            raise RuntimeError("network down")
        return False

    kg_mock.ensure_ontology_imported = AsyncMock(side_effect=ensure_side_effect)
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=2)

    config = _config_with_ontologies("mesh", "go")
    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        results = await backfill_ontology("doc-xyz", config)

    # MeSH ensure_imported raised - import_ok False, no link call.
    assert results["mesh"]["import_ok"] is False
    assert results["mesh"]["n_links"] == 0
    # GO still linked normally.
    assert results["go"]["import_ok"] is True
    assert results["go"]["n_links"] == 2
    # Only one link call (for GO), MeSH was skipped due to import failure.
    kg_mock.link_entities_to_ontology.assert_called_once_with("go", "exact", doc_id="doc-xyz")


async def test_backfill_ontology_empty_when_no_ontology_layers_enabled():
    """No enabled ontologies -> empty result, no client calls."""
    kg_mock = MagicMock(spec=Neo4jClient)
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        results = await backfill_ontology("doc-xyz", config)

    assert results == {}
    kg_mock.ensure_ontology_imported.assert_not_called()
    kg_mock.link_entities_to_ontology.assert_not_called()


# ---- _doc_metadata_fields_from_work helper ----


def test_doc_metadata_fields_from_work_none_returns_eight_null_fields():
    """When work is None, every column is None - shape stays stable."""
    fields = _doc_metadata_fields_from_work(None)
    assert fields == {
        "title": None,
        "year": None,
        "doi": None,
        "openalex_id": None,
        "venue": None,
        "source_url": None,
        "authors_display": None,
        "language": None,
    }


def test_doc_metadata_fields_from_work_populates_from_payload():
    """Helper mirrors the doc-level columns that ingest writes."""
    work = {
        "id": "https://openalex.org/W42",
        "doi": "https://doi.org/10.1/XYZ",
        "title": "Hello Paper",
        "publication_year": 2025,
        "primary_location": {
            "landing_page_url": "http://j.example/x",
            "source": {"display_name": "Journal X"},
        },
        "authorships": [{"author": {"display_name": "Alice"}}],
        "language": "en",
    }
    fields = _doc_metadata_fields_from_work(work)
    assert fields["title"] == "Hello Paper"
    assert fields["year"] == 2025
    assert fields["openalex_id"] == "W42"
    assert fields["doi"] == "10.1/xyz"
    assert fields["venue"] == "Journal X"
    assert fields["source_url"] == "http://j.example/x"
    assert fields["authors_display"] == "Alice"
    assert fields["language"] == "en"


# ---- resolve_openalex ----


def _row_with(**overrides: Any) -> dict[str, Any]:
    """Build a minimal LanceDB row dict for resolve_openalex tests."""
    row = {
        "chunk_id": "docZ#0",
        "doc_id": "docZ",
        "chunk_index": 0,
        "text": "Some DOI here: 10.1/abc",
        "section": None,
        "page": None,
        "content_type": "text",
        "main_label": "Document",
        "sub_label": "Paper",
        "doi": None,
        "metadata_status": "pending",
    }
    row.update(overrides)
    return row


async def test_resolve_openalex_aborts_when_no_chunks_in_lancedb():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    with patch(
        "knowledge_agent.ingestion.metadata_resolution.get_search_client",
        return_value=search_mock,
    ):
        result = await resolve_openalex("docZ")

    assert result["work_resolved"] is False
    assert result["metadata_patched"] is False


async def test_resolve_openalex_skipped_when_manual_and_skip_manual_true():
    """skip_manual=True + status=manual -> no-op, no client calls."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(metadata_status="manual", doi="10.1/manual"),
    ]
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
        ) as rd,
    ):
        result = await resolve_openalex("docZ", skip_manual=True)

    assert result["skipped"] is True
    assert result["work_resolved"] is False
    rd.assert_not_called()


async def test_resolve_openalex_not_skipped_when_manual_and_skip_manual_false():
    """skip_manual=False + status=manual -> proceeds and overwrites."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(metadata_status="manual", doi="10.1/manual"),
    ]
    work = {"id": "W1", "title": "Resolved"}
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await resolve_openalex("docZ", skip_manual=False)

    assert result["skipped"] is False
    assert result["work_resolved"] is True


async def test_resolve_openalex_uses_stored_doi_first():
    """Stored DOI is the fast path; resolve_metadata is not called."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi="10.1/stored"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)

    work = {"id": "W2", "title": "Found"}
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ) as rd,
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_metadata",
            new_callable=AsyncMock,
        ) as rm,
    ):
        result = await resolve_openalex("docZ")

    rd.assert_called_once_with("10.1/stored")
    rm.assert_not_called()  # stored DOI succeeded; fallback skipped
    assert result["work_resolved"] is True


async def test_resolve_openalex_falls_back_to_chunk_extraction_when_stored_doi_fails():
    """Stored DOI doesn't resolve -> fall through to resolve_metadata."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi="10.1/stale"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)

    work = {"id": "W3", "title": "Found Via Extraction"}
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_metadata",
            new_callable=AsyncMock,
            return_value=work,
        ) as rm,
    ):
        result = await resolve_openalex("docZ")

    rm.assert_called_once()
    assert result["work_resolved"] is True


async def test_resolve_openalex_uses_extraction_when_no_stored_doi():
    """No stored doi -> skip resolve_doi, go to resolve_metadata."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi=None),
    ]
    work = {"id": "W4", "title": "From Chunks"}
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=MagicMock()
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
        ) as rd,
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_metadata",
            new_callable=AsyncMock,
            return_value=work,
        ) as rm,
    ):
        await resolve_openalex("docZ")

    rd.assert_not_called()  # no stored DOI to try
    rm.assert_called_once()


async def test_resolve_openalex_no_work_resolved_leaves_state_untouched():
    """Nothing resolves -> no LanceDB patch, no KG writes."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[_row_with()])
    kg_mock = MagicMock(spec=Neo4jClient)
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_metadata",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await resolve_openalex("docZ")

    assert result["work_resolved"] is False
    assert result["metadata_patched"] is False
    assert result["kg_l1_l4_ok"] is False
    assert result["new_status"] is None
    search_mock.update_doc_metadata.assert_not_called()
    kg_mock.delete_doc_l1_l4_edges.assert_not_called()


async def test_resolve_openalex_paper_path_does_surgical_wipe_then_writes():
    """Successful resolve for Paper -> delete_doc_l1_l4_edges + 4 writes."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi="10.1/x", sub_label="Paper"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)
    work = {"id": "W5", "title": "OK"}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await resolve_openalex("docZ")

    assert result["kg_l1_l4_ok"] is True
    # Surgical wipe used - NOT delete_doc.
    kg_mock.delete_doc_l1_l4_edges.assert_called_once_with("docZ")
    kg_mock.delete_doc.assert_not_called()
    kg_mock.write_citations.assert_called_once_with("docZ", work)
    kg_mock.write_authorships.assert_called_once_with("docZ", work)
    kg_mock.write_venue.assert_called_once_with("docZ", work)
    kg_mock.write_topics.assert_called_once_with("docZ", work)


async def test_resolve_openalex_non_paper_patches_lancedb_only_skips_kg_writes():
    """sub_label != Paper -> LanceDB patched, no KG L1-L4 writes."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi="10.1/x", sub_label="Note"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    work = {"id": "W6", "title": "Note resolved"}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await resolve_openalex("docZ")

    assert result["metadata_patched"] is True
    assert result["kg_l1_l4_ok"] is False  # KG L1-L4 not applicable
    search_mock.update_doc_metadata.assert_called_once()
    kg_mock.delete_doc_l1_l4_edges.assert_not_called()
    kg_mock.write_citations.assert_not_called()


async def test_resolve_openalex_patches_with_enriched_status():
    """metadata_status in the patch dict becomes 'enriched' after success."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(doi="10.1/x", metadata_status="pending"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)
    work = {"id": "W7", "title": "Resolved", "publication_year": 2026}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await resolve_openalex("docZ")

    assert result["new_status"] == "enriched"
    # LanceDB patch fields include enriched status + the work-derived cols.
    args, _ = search_mock.update_doc_metadata.call_args
    doc_id, fields = args
    assert doc_id == "docZ"
    assert fields["metadata_status"] == "enriched"
    assert fields["title"] == "Resolved"
    assert fields["year"] == 2026
    assert fields["openalex_id"] == "W7"


# ---- lookup_known_doi ----


async def test_lookup_known_doi_empty_doi_returns_failure():
    """Empty DOI is a programming error - return failure, no client calls."""
    search_mock = MagicMock(spec=LanceClient)
    with patch(
        "knowledge_agent.ingestion.metadata_resolution.get_search_client",
        return_value=search_mock,
    ):
        result = await lookup_known_doi("docZ", "")
    assert result["work_resolved"] is False
    search_mock.get_chunks_by_doc_id.assert_not_called()


async def test_lookup_known_doi_no_chunks_returns_failure():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    with patch(
        "knowledge_agent.ingestion.metadata_resolution.get_search_client",
        return_value=search_mock,
    ):
        result = await lookup_known_doi("docZ", "10.1/x")
    assert result["work_resolved"] is False


async def test_lookup_known_doi_never_falls_back_to_chunk_extraction():
    """resolve_doi returning None -> failure, NOT a resolve_metadata call."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[_row_with()])
    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_metadata",
            new_callable=AsyncMock,
        ) as rm,
    ):
        result = await lookup_known_doi("docZ", "10.1/nonexistent")

    assert result["work_resolved"] is False
    # Critical contract: NO fallback to chunk-text extraction. The user
    # typed this DOI explicitly; we don't second-guess it by extracting
    # a different one from chunks.
    rm.assert_not_called()


async def test_lookup_known_doi_happy_path_patches_lancedb_and_kg():
    """Successful resolve -> LanceDB patch + surgical KG L1-L4 rewrite."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(sub_label="Paper", doi=None),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)
    work = {"id": "W11", "title": "User-Supplied"}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ) as rd,
    ):
        result = await lookup_known_doi("docZ", "10.1/typed-by-user")

    rd.assert_called_once_with("10.1/typed-by-user")
    assert result["work_resolved"] is True
    assert result["metadata_patched"] is True
    assert result["kg_l1_l4_ok"] is True
    assert result["new_status"] == "enriched"
    kg_mock.delete_doc_l1_l4_edges.assert_called_once_with("docZ")


async def test_lookup_known_doi_non_paper_patches_lancedb_only():
    """Non-Paper sub_label -> LanceDB patched, KG L1-L4 skipped."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(sub_label="Note"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    work = {"id": "W12", "title": "Note"}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await lookup_known_doi("docZ", "10.1/note")

    assert result["metadata_patched"] is True
    assert result["kg_l1_l4_ok"] is False
    kg_mock.delete_doc_l1_l4_edges.assert_not_called()


async def test_lookup_known_doi_has_no_skip_manual_concept():
    """Even manual-edited docs proceed (clicking the DOI button is consent)."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        _row_with(metadata_status="manual", sub_label="Paper"),
    ]
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_citations = AsyncMock(return_value=True)
    kg_mock.write_authorships = AsyncMock(return_value=True)
    kg_mock.write_venue = AsyncMock(return_value=True)
    kg_mock.write_topics = AsyncMock(return_value=True)
    work = {"id": "W13", "title": "Replaced manual"}

    with (
        patch(
            "knowledge_agent.ingestion.metadata_resolution.get_search_client",
            return_value=search_mock,
        ),
        patch("knowledge_agent.ingestion.metadata_resolution.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.metadata_resolution.resolve_doi",
            new_callable=AsyncMock,
            return_value=work,
        ),
    ):
        result = await lookup_known_doi("docZ", "10.1/new")

    assert result["work_resolved"] is True
    assert result["metadata_patched"] is True


# ---- re_embed ----


def _re_embed_config(optimize: bool = False) -> Any:
    """Build a minimal MagicMock CorpusConfig for re_embed tests.

    Only `optimize_indexes_per_ingest` matters — that's the sole field
    `re_embed` reads. Split out because 6 tests need the same setup.
    """
    cfg = MagicMock()
    cfg.optimize_indexes_per_ingest = optimize
    return cfg


async def test_re_embed_aborts_when_lancedb_read_fails():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    with patch(
        "knowledge_agent.ingestion.pipeline.get_search_client",
        return_value=search_mock,
    ):
        result = await re_embed("docZ", _re_embed_config())

    assert result == {"embed_ok": False, "lancedb_ok": False, "n_chunks": 0}


async def test_re_embed_skips_when_no_chunks_found():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    with patch(
        "knowledge_agent.ingestion.pipeline.get_search_client",
        return_value=search_mock,
    ):
        result = await re_embed("docZ", _re_embed_config())

    assert result == {"embed_ok": False, "lancedb_ok": False, "n_chunks": 0}


async def test_re_embed_returns_embed_false_when_embedder_fails():
    """embed_texts raising -> embed_ok False; no LanceDB writes.
    Under the typed-errors contract embed_texts raises on Voyage
    failure; re_embed's wrapper converts that into the dict signal."""
    rows = [
        {
            "chunk_id": "docZ#0",
            "doc_id": "docZ",
            "chunk_index": 0,
            "text": "alpha",
            "embedding": [0.0] * 4,
        },
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.embed_chunks",
            new_callable=AsyncMock,
            side_effect=RuntimeError("voyage boom"),
        ),
    ):
        result = await re_embed("docZ", _re_embed_config())

    assert result["embed_ok"] is False
    assert result["lancedb_ok"] is False
    assert result["n_chunks"] == 1
    search_mock.delete_chunks_by_doc_id.assert_not_called()
    search_mock.write_chunks.assert_not_called()


async def test_re_embed_happy_path_swaps_embeddings_and_rewrites():
    """Reads rows, re-embeds text, mutates embeddings, delete+write."""
    rows = [
        {
            "chunk_id": "docZ#0",
            "doc_id": "docZ",
            "chunk_index": 0,
            "text": "alpha",
            "embedding": [0.1, 0.2],
            "title": "old",
        },
        {
            "chunk_id": "docZ#1",
            "doc_id": "docZ",
            "chunk_index": 1,
            "text": "beta",
            "embedding": [0.3, 0.4],
            "title": "old",
        },
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    search_mock.write_chunks = AsyncMock(return_value=True)

    new_vecs = [[0.9, 0.8], [0.7, 0.6]]

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.embed_chunks",
            new_callable=AsyncMock,
            return_value=new_vecs,
        ),
    ):
        result = await re_embed("docZ", _re_embed_config(optimize=False))

    assert result["embed_ok"] is True
    assert result["lancedb_ok"] is True
    assert result["n_chunks"] == 2
    # Old rows wiped, new rows written.
    search_mock.delete_chunks_by_doc_id.assert_called_once_with("docZ")
    write_args, _ = search_mock.write_chunks.call_args
    written = write_args[0]
    # Embeddings swapped in place; other fields preserved.
    assert written[0]["embedding"] == [0.9, 0.8]
    assert written[1]["embedding"] == [0.7, 0.6]
    assert written[0]["title"] == "old"  # doc-level fields preserved
    # No index rebuild because the config setting was False.
    search_mock.ensure_indexes.assert_not_called()


async def test_re_embed_rebuilds_index_when_setting_enabled():
    rows = [
        {
            "chunk_id": "docZ#0",
            "doc_id": "docZ",
            "chunk_index": 0,
            "text": "alpha",
            "embedding": [0.0],
        },
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    search_mock.write_chunks = AsyncMock(return_value=True)

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.embed_chunks",
            new_callable=AsyncMock,
            return_value=[[1.0]],
        ),
    ):
        await re_embed("docZ", _re_embed_config(optimize=True))

    search_mock.ensure_indexes.assert_called_once()


async def test_re_embed_skips_index_rebuild_when_lancedb_write_fails():
    """Don't waste an index rebuild on a doc whose write didn't land.

    write_chunks raising under the typed-errors contract flips
    `lancedb_ok=False` inside `re_embed`, which gates the index call.
    """
    rows = [
        {"chunk_id": "docZ#0", "doc_id": "docZ", "chunk_index": 0, "text": "a", "embedding": [0.0]},
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    search_mock.write_chunks = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.embed_chunks",
            new_callable=AsyncMock,
            return_value=[[1.0]],
        ),
    ):
        await re_embed("docZ", _re_embed_config(optimize=True))

    search_mock.ensure_indexes.assert_not_called()


# ---- backfill_chunks ----


def _chunk_row(
    chunk_index: int,
    text: str,
    main_label: str = "Document",
    sub_label: str | None = "Paper",
    section: str | None = None,
    page: int | None = None,
    content_type: str = "text",
) -> dict[str, Any]:
    """Build one LanceDB-shaped chunk row dict for backfill tests."""
    return {
        "chunk_id": f"docZ#{chunk_index}",
        "doc_id": "docZ",
        "chunk_index": chunk_index,
        "text": text,
        "main_label": main_label,
        "sub_label": sub_label,
        "section": section,
        "page": page,
        "content_type": content_type,
    }


async def test_backfill_chunks_no_op_when_layer_disabled():
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=False),
    )
    search_mock = MagicMock(spec=LanceClient)
    kg_mock = MagicMock(spec=Neo4jClient)
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_chunks("docZ", config)

    assert result == {"chunks_ok": False, "entities": {}}
    search_mock.get_chunks_by_doc_id.assert_not_called()


async def test_backfill_chunks_aborts_when_lancedb_read_fails():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_chunks("docZ", config)

    assert result["chunks_ok"] is False
    kg_mock.delete_chunks_by_doc_id.assert_not_called()
    kg_mock.write_chunks.assert_not_called()


async def test_backfill_chunks_skips_when_no_chunks_found():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_chunks("docZ", config)

    assert result == {"chunks_ok": False, "entities": {}}
    kg_mock.write_chunks.assert_not_called()


async def test_backfill_chunks_happy_path_rewrites_kg_and_chains_into_entities():
    rows = [
        _chunk_row(0, "hello", section="Intro", page=1),
        _chunk_row(1, "world", section="Methods", page=2),
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_chunks = AsyncMock(return_value=True)
    kg_mock.write_entities = AsyncMock(return_value=True)
    extractor_mock = AsyncMock(return_value=[])

    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_chunks("docZ", config)

    assert result["chunks_ok"] is True
    # Old KG chunks wiped first.
    kg_mock.delete_chunks_by_doc_id.assert_called_once_with("docZ")
    # write_chunks called with reconstructed ParsedChunks + recovered labels.
    args, _ = kg_mock.write_chunks.call_args
    doc_id, chunks, main_label, sub_label = args
    assert doc_id == "docZ"
    assert main_label == "Document"
    assert sub_label == "Paper"
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.text for c in chunks] == ["hello", "world"]
    assert chunks[0].section == "Intro"
    assert chunks[0].page == 1
    # Entities chain ran (entities layer enabled in fixture config).
    assert "entities_ok" in result["entities"]


async def test_backfill_chunks_recovers_labels_from_first_row():
    """main_label/sub_label come from row[0] (all rows share these fields)."""
    rows = [
        _chunk_row(0, "x", main_label="Artifact", sub_label="Dataset"),
        _chunk_row(1, "y", main_label="Artifact", sub_label="Dataset"),
    ]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_chunks = AsyncMock(return_value=True)
    config = CorpusConfig(
        allowed_types=["Dataset"],
        layers=LayerFlags(chunks=True),
    )

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        await backfill_chunks("docZ", config)

    args, _ = kg_mock.write_chunks.call_args
    _, _, main_label, sub_label = args
    assert main_label == "Artifact"
    assert sub_label == "Dataset"


async def test_backfill_chunks_skips_entities_when_kg_write_fails():
    """write_chunks raising -> orchestrator records chunks_ok=False and
    doesn't propagate downstream on a known-broken layer."""
    rows = [_chunk_row(0, "a")]
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_chunks = AsyncMock(side_effect=RuntimeError("boom"))
    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_chunks("docZ", config)

    assert result["chunks_ok"] is False
    assert result["entities"] == {}
    kg_mock.delete_entities_by_doc_id.assert_not_called()


# ---- backfill_entities ----


def _entities_enabled_config(*ontology_names: str) -> CorpusConfig:
    """CorpusConfig with chunks + entities + optional ontologies enabled."""
    flags_kwargs = {"chunks": True, "entities": True}
    ontology_cfg = {}
    for n in ontology_names:
        flags_kwargs[f"ontology_{n}"] = True
        ontology_cfg[n] = OntologyConfig(matching="exact")
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(**flags_kwargs),
        entities=EntityConfig(extractor="llm", entity_types=["GENE", "DISEASE"]),
        ontology=ontology_cfg,
    )


async def test_backfill_entities_no_op_when_layer_disabled():
    """entities=False -> immediate return, no client calls."""
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),
    )
    search_mock = MagicMock(spec=LanceClient)
    kg_mock = MagicMock(spec=Neo4jClient)
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result == {"entities_ok": False, "n_mentions": 0, "ontology": {}}
    search_mock.get_chunks_by_doc_id.assert_not_called()
    kg_mock.delete_entities_by_doc_id.assert_not_called()


async def test_backfill_entities_aborts_when_lancedb_read_fails():
    """LanceDB None -> abort, no KG side effects."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is False
    kg_mock.delete_entities_by_doc_id.assert_not_called()
    kg_mock.write_entities.assert_not_called()


async def test_backfill_entities_skips_when_no_chunks_found():
    """Doc has no chunks in LanceDB -> nothing to backfill."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()
    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is False
    assert result["n_mentions"] == 0
    kg_mock.delete_entities_by_doc_id.assert_not_called()


async def test_backfill_entities_happy_path_extracts_and_writes():
    """Reads chunks, deletes old entities, writes new, returns mention count."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha", "chunk_index": 0},
        {"chunk_id": "doc-1#1", "text": "beta", "chunk_index": 1},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)

    # Two mentions in chunk 0, one in chunk 1.
    extractor_mock = AsyncMock(
        side_effect=[
            [
                Mention(raw_text="A", entity_type="GENE", offset=0, confidence=None),
                Mention(raw_text="B", entity_type="GENE", offset=2, confidence=None),
            ],
            [Mention(raw_text="C", entity_type="DISEASE", offset=0, confidence=None)],
        ]
    )
    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is True
    assert result["n_mentions"] == 3
    # Old entities wiped before new write.
    kg_mock.delete_entities_by_doc_id.assert_called_once_with("doc-1")
    # write_entities received the full chunk-mentions list.
    args, _ = kg_mock.write_entities.call_args
    assert args[0] == "doc-1"
    chunk_mentions = args[1]
    assert [cm[0] for cm in chunk_mentions] == ["doc-1#0", "doc-1#1"]
    assert len(chunk_mentions[0][1]) == 2
    assert len(chunk_mentions[1][1]) == 1


async def test_backfill_entities_one_chunk_extraction_failure_does_not_poison_others():
    """Extractor raising on one chunk -> that chunk gets [], others run normally."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha", "chunk_index": 0},
        {"chunk_id": "doc-1#1", "text": "beta", "chunk_index": 1},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)

    # First chunk extraction raises; second succeeds.
    extractor_mock = AsyncMock(
        side_effect=[
            RuntimeError("model down"),
            [Mention(raw_text="X", entity_type="GENE", offset=0, confidence=None)],
        ]
    )
    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is True
    # Only chunk 1 produced a mention; chunk 0 contributed 0.
    assert result["n_mentions"] == 1
    args, _ = kg_mock.write_entities.call_args
    chunk_mentions = args[1]
    assert chunk_mentions[0][1] == []  # chunk 0 mentions empty
    assert len(chunk_mentions[1][1]) == 1


async def test_backfill_entities_chains_into_backfill_ontology_when_entities_ok():
    """Happy path with ontology enabled -> backfill_ontology runs after write."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha", "chunk_index": 0},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=False)
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=5)

    extractor_mock = AsyncMock(
        return_value=[
            Mention(raw_text="A", entity_type="GENE", offset=0, confidence=None),
        ]
    )

    config = _entities_enabled_config("mesh")

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["ontology"] == {
        "mesh": {"imported": True, "import_ok": True, "n_links": 5},
    }
    kg_mock.link_entities_to_ontology.assert_called_once_with("mesh", "exact", doc_id="doc-1")


async def test_backfill_entities_skips_ontology_when_entity_write_fails():
    """write_entities raising -> orchestrator records entities_ok=False
    and doesn't run ontology linking on stale state."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha", "chunk_index": 0},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(side_effect=RuntimeError("boom"))

    extractor_mock = AsyncMock(return_value=[])
    config = _entities_enabled_config("mesh")

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is False
    assert result["ontology"] == {}
    kg_mock.ensure_ontology_imported.assert_not_called()


# ---- backfill_triples (L8 per-doc) ----


def _triples_enabled_config() -> CorpusConfig:
    """Minimal L8-enabled config: chunks + entities + triples."""
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, triples=True),
        entities=EntityConfig(extractor="llm", entity_types=["GENE", "DISEASE"]),
    )


async def test_backfill_triples_no_op_when_layer_disabled():
    """triples=False -> immediate return, no client calls."""
    search_mock = MagicMock(spec=LanceClient)
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()  # entities on, triples off

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_triples("doc-x", config)

    assert result == {"triples_ok": False, "n_triples": 0}
    search_mock.get_chunks_by_doc_id.assert_not_called()
    kg_mock.write_triples.assert_not_called()


async def test_backfill_triples_aborts_when_lancedb_read_fails():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _triples_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_triples("doc-x", config)

    assert result == {"triples_ok": False, "n_triples": 0}
    kg_mock.delete_triples_by_doc_id.assert_not_called()
    kg_mock.write_triples.assert_not_called()


async def test_backfill_triples_skips_when_no_chunks_found():
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _triples_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        result = await backfill_triples("doc-x", config)

    assert result == {"triples_ok": False, "n_triples": 0}
    kg_mock.write_triples.assert_not_called()


async def test_backfill_triples_happy_path_extracts_and_writes():
    """Chunks + entities present -> LLM extracts -> write_triples called."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "BRCA1 inhibits TP53."},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.get_entities_by_chunk.return_value = {
        "doc-1#0": [("brca1", "GENE"), ("tp53", "GENE")],
    }
    kg_mock.write_triples = AsyncMock(return_value=True)

    triple = ExtractedTriple(
        subject_key="brca1",
        subject_entity_type="GENE",
        predicate="INHIBITS",
        object_key="tp53",
        object_entity_type="GENE",
        evidence_span="BRCA1 inhibits TP53.",
    )
    config = _triples_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.triples_extractor.extract", return_value=[triple]
        ) as mock_extract,
    ):
        result = await backfill_triples("doc-1", config)

    assert result["triples_ok"] is True
    assert result["n_triples"] == 1
    # Stale wipe ran before write.
    kg_mock.delete_triples_by_doc_id.assert_called_once_with("doc-1")
    kg_mock.write_triples.assert_called_once_with("doc-1", [("doc-1#0", [triple])])
    # Extractor got the chunk text + vocab, plus per-corpus model +
    # temperature (2026-07-02 refactor: no more settings global read).
    mock_extract.assert_called_once_with(
        "BRCA1 inhibits TP53.",
        [("brca1", "GENE"), ("tp53", "GENE")],
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
    )


async def test_backfill_triples_skips_chunk_with_empty_vocab():
    """Chunk has no L6 entities -> no LLM call, but doc still written."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "Methodology overview."},
        {"chunk_id": "doc-1#1", "text": "BRCA1 inhibits TP53."},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    # First chunk has no vocab; second has two entities.
    kg_mock.get_entities_by_chunk.return_value = {
        "doc-1#1": [("brca1", "GENE"), ("tp53", "GENE")],
    }
    kg_mock.write_triples = AsyncMock(return_value=True)

    config = _triples_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.triples_extractor.extract", return_value=[]
        ) as mock_extract,
    ):
        result = await backfill_triples("doc-1", config)

    assert result["triples_ok"] is True
    # Only ONE LLM call - the empty-vocab chunk was skipped.
    assert mock_extract.call_count == 1
    # Both chunks appear in the write payload (the empty one as
    # `(chunk_id, [])` for completeness).
    args, _ = kg_mock.write_triples.call_args
    payload = args[1]
    chunk_ids = [c for c, _ in payload]
    assert chunk_ids == ["doc-1#0", "doc-1#1"]


async def test_backfill_triples_one_chunk_extraction_failure_does_not_poison_others():
    """LLM raises on chunk #0 -> log + skip + continue to chunk #1."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "bad chunk"},
        {"chunk_id": "doc-1#1", "text": "good chunk"},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.get_entities_by_chunk.return_value = {
        "doc-1#0": [("a", "GENE"), ("b", "GENE")],
        "doc-1#1": [("c", "GENE"), ("d", "GENE")],
    }
    kg_mock.write_triples = AsyncMock(return_value=True)
    config = _triples_enabled_config()

    good_triple = ExtractedTriple(
        subject_key="c",
        subject_entity_type="GENE",
        predicate="INHIBITS",
        object_key="d",
        object_entity_type="GENE",
        evidence_span="c inhibits d",
    )

    def fake_extract(text, vocab, *, model, temperature):
        if "bad" in text:
            raise RuntimeError("boom")
        return [good_triple]

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch(
            "knowledge_agent.ingestion.pipeline.triples_extractor.extract", side_effect=fake_extract
        ),
    ):
        result = await backfill_triples("doc-1", config)

    # The doc still succeeded; n_triples counts only the good chunk.
    assert result["triples_ok"] is True
    assert result["n_triples"] == 1


async def test_backfill_entities_chains_into_backfill_triples_when_triples_enabled():
    """Propagation: backfill_entities -> backfill_triples (when triples on)."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha"},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)
    # backfill_triples re-reads chunks + entities from these mocks.
    kg_mock.get_entities_by_chunk.return_value = {
        "doc-1#0": [("alpha", "GENE")],
    }
    kg_mock.write_triples = AsyncMock(return_value=True)

    extractor_mock = AsyncMock(return_value=[Mention(raw_text="alpha", entity_type="GENE")])

    config = _triples_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
        patch("knowledge_agent.ingestion.pipeline.triples_extractor.extract", return_value=[]),
    ):
        result = await backfill_entities("doc-1", config)

    # backfill_entities chained into backfill_triples; both write paths
    # ran.
    assert result["entities_ok"] is True
    assert "triples" in result
    assert result["triples"] == {"triples_ok": True, "n_triples": 0}
    kg_mock.write_triples.assert_called_once()


async def test_backfill_entities_does_not_run_triples_when_triples_off():
    """triples=False but entities on -> ontology still runs but triples
    sub-result stays empty (no-op return)."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha"},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)
    extractor_mock = AsyncMock(return_value=[])
    # triples disabled in this config.
    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    # triples short-circuited at the layer-disabled check.
    assert result["triples"] == {"triples_ok": False, "n_triples": 0}
    kg_mock.write_triples.assert_not_called()


# ---- pipeline.delete_doc must wipe triples before entity GC ----


async def test_delete_doc_calls_delete_triples_before_delete_entities():
    """Order matters: triples reference :Entity nodes via Cypher edges;
    if the entity-orphan GC runs first while triples still point at
    those entities, the plain DELETE crashes. Verify the call order."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.delete_chunks_by_doc_id = AsyncMock(return_value=True)
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.delete_doc = AsyncMock(return_value=True)
    kg_mock.delete_chunks_by_doc_id = AsyncMock(return_value=True)
    kg_mock.delete_triples_by_doc_id = AsyncMock(return_value=True)
    kg_mock.delete_entities_by_doc_id = AsyncMock(return_value=True)

    # Record the order of calls across both mocks via a single Mock.
    call_order = []
    kg_mock.delete_doc = AsyncMock(
        side_effect=lambda *a, **kw: call_order.append("delete_doc") or True
    )
    kg_mock.delete_chunks_by_doc_id.side_effect = lambda *a, **kw: (
        call_order.append("delete_chunks") or True
    )
    kg_mock.delete_triples_by_doc_id.side_effect = lambda *a, **kw: (
        call_order.append("delete_triples") or True
    )
    kg_mock.delete_entities_by_doc_id.side_effect = lambda *a, **kw: (
        call_order.append("delete_entities") or True
    )

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
    ):
        ok = await delete_doc("doc-1")

    assert ok is True
    # The crucial ordering: triples wiped BEFORE entity GC.
    triples_idx = call_order.index("delete_triples")
    entities_idx = call_order.index("delete_entities")
    assert triples_idx < entities_idx


# ---- IngestResult L8 fields ----


def test_ingest_result_l8_fields_default_to_safe_values():
    """IngestResult's L8 fields default to (False, 0) so a doc that
    didn't run L8 surfaces as cleanly absent rather than indeterminate."""
    result = IngestResult(
        doc_id="x",
        path=Path("/tmp/x"),
        n_chunks=0,
        metadata_status="baseline",
        work=None,
        embed_ok=False,
        embed_error=None,
        lancedb_ok=False,
        lancedb_error=None,
        kg_citations_ok=False,
        kg_citations_error=None,
        kg_authorships_ok=False,
        kg_authorships_error=None,
        kg_venue_ok=False,
        kg_venue_error=None,
        kg_topics_ok=False,
        kg_topics_error=None,
        kg_chunks_ok=False,
        kg_chunks_error=None,
        kg_entities_ok=False,
        kg_entities_error=None,
        n_entity_mentions=0,
    )
    assert result.kg_triples_ok is False
    assert result.n_triples_written == 0


# ---- backfill_cross_doc (L9 per-doc) ----


def _cross_doc_enabled_config() -> CorpusConfig:
    """Minimal L9-enabled config: chunks + entities + cross_doc."""
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, cross_doc=True),
        entities=EntityConfig(extractor="llm", entity_types=["GENE", "DISEASE"]),
    )


async def test_backfill_cross_doc_no_op_when_layer_disabled():
    """cross_doc=False -> immediate return, no client calls."""
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()  # entities on, cross_doc off

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc("doc-x", config)

    assert result == {"cross_doc_ok": False, "n_edges": 0}
    kg_mock.recompute_cross_doc_edges.assert_not_called()


async def test_backfill_cross_doc_happy_path_returns_count():
    """recompute returns int -> ok=True, n=that int."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_edges = AsyncMock(return_value=4)
    config = _cross_doc_enabled_config()

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc("doc-1", config)

    assert result == {"cross_doc_ok": True, "n_edges": 4}
    kg_mock.recompute_cross_doc_edges.assert_called_once_with("doc-1", 2)


async def test_backfill_cross_doc_zero_edges_is_success():
    """No other doc met threshold -> ok=True, n=0 (distinct from
    cross_doc_ok=False which means Cypher failure)."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_edges = AsyncMock(return_value=0)
    config = _cross_doc_enabled_config()

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc("doc-1", config)

    assert result == {"cross_doc_ok": True, "n_edges": 0}


async def test_backfill_cross_doc_exception_means_failure():
    """recompute raising -> ok=False, n=0 (orchestrator catches)."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_edges = AsyncMock(side_effect=RuntimeError("boom"))
    config = _cross_doc_enabled_config()

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc("doc-1", config)

    assert result == {"cross_doc_ok": False, "n_edges": 0}


async def test_backfill_entities_chains_into_backfill_cross_doc_when_enabled():
    """Propagation: backfill_entities -> backfill_cross_doc (when L9 on)."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha"},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)
    kg_mock.recompute_cross_doc_edges = AsyncMock(return_value=3)

    extractor_mock = AsyncMock(return_value=[Mention(raw_text="alpha", entity_type="GENE")])

    config = _cross_doc_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["entities_ok"] is True
    assert result["cross_doc"] == {"cross_doc_ok": True, "n_edges": 3}
    kg_mock.recompute_cross_doc_edges.assert_called_once_with("doc-1", 2)


async def test_backfill_entities_does_not_run_cross_doc_when_disabled():
    """L9 off -> cross_doc sub-result is the no-op shape."""
    search_mock = MagicMock(spec=LanceClient)
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc-1#0", "text": "alpha"},
    ]
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.write_entities = AsyncMock(return_value=True)
    extractor_mock = AsyncMock(return_value=[])
    # cross_doc disabled.
    config = _entities_enabled_config()

    with (
        patch("knowledge_agent.ingestion.pipeline.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.pipeline.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.pipeline.extract_union", extractor_mock),
    ):
        result = await backfill_entities("doc-1", config)

    assert result["cross_doc"] == {"cross_doc_ok": False, "n_edges": 0}
    kg_mock.recompute_cross_doc_edges.assert_not_called()


def test_ingest_result_l9_fields_default_to_safe_values():
    """L9 fields default to (False, 0) so a doc that didn't run L9
    surfaces as cleanly absent."""
    result = IngestResult(
        doc_id="x",
        path=Path("/tmp/x"),
        n_chunks=0,
        metadata_status="baseline",
        work=None,
        embed_ok=False,
        embed_error=None,
        lancedb_ok=False,
        lancedb_error=None,
        kg_citations_ok=False,
        kg_citations_error=None,
        kg_authorships_ok=False,
        kg_authorships_error=None,
        kg_venue_ok=False,
        kg_venue_error=None,
        kg_topics_ok=False,
        kg_topics_error=None,
        kg_chunks_ok=False,
        kg_chunks_error=None,
        kg_entities_ok=False,
        kg_entities_error=None,
        n_entity_mentions=0,
    )
    assert result.kg_cross_doc_ok is False
    assert result.n_cross_doc_edges_written == 0


# ---- _build_lance_rows ----


def test_build_lance_rows_with_work_populates_denorm_cache():
    chunks = [_chunk(0, "hello", "Intro", 1)]
    embeddings = [[0.1, 0.2]]
    work = {
        "id": "https://openalex.org/W1234",
        "doi": "https://doi.org/10.1/ABC",
        "title": "Test Paper",
        "publication_year": 2024,
        "primary_location": {
            "landing_page_url": "http://example.com/p",
            "source": {"display_name": "Journal X"},
        },
        "authorships": [{"author": {"display_name": "Alice"}}],
        "language": "en",
    }
    rows = _build_lance_rows(
        "docabc",
        chunks,
        embeddings,
        work,
        "enriched",
        "Document",
        "Paper",
        Path("/tmp/paper.pdf"),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["chunk_id"] == "docabc#0"
    assert row["doc_id"] == "docabc"
    assert row["chunk_index"] == 0
    assert row["text"] == "hello"
    assert row["section"] == "Intro"
    assert row["page"] == 1
    assert row["embedding"] == [0.1, 0.2]
    assert row["main_label"] == "Document"
    assert row["sub_label"] == "Paper"
    # DOI normalised (lowercased, URL prefix stripped).
    assert row["doi"] == "10.1/abc"
    # OpenAlex ID extracted to bare form.
    assert row["openalex_id"] == "W1234"
    assert row["title"] == "Test Paper"
    assert row["year"] == 2024
    assert row["authors_display"] == "Alice"
    assert row["venue"] == "Journal X"
    assert row["source_url"] == "http://example.com/p"
    assert row["metadata_status"] == "enriched"
    assert row["language"] == "en"
    # source_path captured via as_posix() so storage uses forward
    # slashes on every OS - lets sync compare stored vs current cleanly.
    assert row["source_path"] == "/tmp/paper.pdf"
    assert isinstance(row["ingested_at"], datetime)


def test_build_lance_rows_without_work_leaves_metadata_none():
    chunks = [_chunk(0, "hello")]
    embeddings = [[0.0]]
    rows = _build_lance_rows(
        "docabc",
        chunks,
        embeddings,
        None,
        "baseline",
        "Document",
        None,
        Path("/tmp/loose.txt"),
    )
    row = rows[0]
    assert row["doi"] is None
    assert row["openalex_id"] is None
    assert row["title"] is None
    assert row["year"] is None
    assert row["authors_display"] is None
    assert row["venue"] is None
    assert row["source_url"] is None
    assert row["language"] is None
    assert row["metadata_status"] == "baseline"
    # main_label always set, sub_label allowed to be None.
    assert row["main_label"] == "Document"
    assert row["sub_label"] is None
    # Required structural fields still populated.
    assert row["chunk_id"] == "docabc#0"
    assert row["doc_id"] == "docabc"
    assert row["text"] == "hello"
    # source_path is always captured (via as_posix) even when work is absent.
    assert row["source_path"] == "/tmp/loose.txt"


def test_build_lance_rows_uses_make_chunk_id_pattern():
    chunks = [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")]
    embeddings = [[0.0], [0.0], [0.0]]
    rows = _build_lance_rows(
        "doc1",
        chunks,
        embeddings,
        None,
        "baseline",
        "Document",
        None,
        Path("/tmp/x.pdf"),
    )
    chunk_ids = [r["chunk_id"] for r in rows]
    assert chunk_ids == ["doc1#0", "doc1#1", "doc1#2"]


def test_build_lance_rows_zip_strict_raises_on_length_mismatch():
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    embeddings = [[0.0]]  # 1 vector for 2 chunks
    with pytest.raises(ValueError):
        _build_lance_rows(
            "doc",
            chunks,
            embeddings,
            None,
            "baseline",
            "Document",
            None,
            Path("/tmp/x.pdf"),
        )


def test_build_lance_rows_handles_missing_primary_location():
    chunks = [_chunk(0, "hello")]
    embeddings = [[0.0]]
    work = {"id": "https://openalex.org/W1", "title": "Test"}
    rows = _build_lance_rows(
        "doc",
        chunks,
        embeddings,
        work,
        "enriched",
        "Document",
        "Paper",
        Path("/tmp/x.pdf"),
    )
    assert rows[0]["venue"] is None
    assert rows[0]["source_url"] is None


# ---- IngestResult ----


def test_ingest_result_dataclass_fields():
    r = IngestResult(
        doc_id="abc",
        path=Path("/tmp/file.pdf"),
        n_chunks=10,
        metadata_status="enriched",
        work={"id": "W1"},
        embed_ok=True,
        embed_error=None,
        lancedb_ok=True,
        lancedb_error=None,
        kg_citations_ok=True,
        kg_citations_error=None,
        kg_authorships_ok=True,
        kg_authorships_error=None,
        kg_venue_ok=True,
        kg_venue_error=None,
        kg_topics_ok=True,
        kg_topics_error=None,
        kg_chunks_ok=True,
        kg_chunks_error=None,
        kg_entities_ok=True,
        kg_entities_error=None,
        n_entity_mentions=42,
    )
    assert r.doc_id == "abc"
    assert r.n_chunks == 10
    assert r.metadata_status == "enriched"
    assert r.embed_ok is True
    assert r.kg_venue_ok is True
    assert r.kg_topics_ok is True
    assert r.kg_chunks_ok is True
    assert r.kg_entities_ok is True
    assert r.n_entity_mentions == 42


def test_ingest_result_tracks_kg_layers_independently():
    """Each KG layer write succeeds/fails independently of the others."""
    r = IngestResult(
        doc_id="abc",
        path=Path("/tmp/file.pdf"),
        n_chunks=5,
        metadata_status="enriched",
        work={"id": "W1"},
        embed_ok=True,
        embed_error=None,
        lancedb_ok=True,
        lancedb_error=None,
        kg_citations_ok=True,
        kg_citations_error=None,
        kg_authorships_ok=True,
        kg_authorships_error=None,
        kg_venue_ok=False,  # venue write failed
        kg_venue_error=None,
        kg_topics_ok=True,
        kg_topics_error=None,
        kg_chunks_ok=True,
        kg_chunks_error=None,
        kg_entities_ok=False,  # extractor backend tripped
        kg_entities_error=None,
        n_entity_mentions=0,
    )
    assert r.kg_citations_ok is True
    assert r.kg_authorships_ok is True
    assert r.kg_venue_ok is False
    assert r.kg_topics_ok is True
    assert r.kg_chunks_ok is True
    assert r.kg_entities_ok is False


def test_ingest_result_kg_chunks_independent_of_openalex_layers():
    """kg_chunks_ok can be True even when no OpenAlex data resolved -
    chunks are written from docling parsing, not from `work`."""
    r = IngestResult(
        doc_id="abc",
        path=Path("/tmp/file.pdf"),
        n_chunks=5,
        metadata_status="baseline",  # no OpenAlex resolution
        work=None,
        embed_ok=True,
        embed_error=None,
        lancedb_ok=True,
        lancedb_error=None,
        kg_citations_ok=False,  # not run - no work
        kg_citations_error=None,
        kg_authorships_ok=False,
        kg_authorships_error=None,
        kg_venue_ok=False,
        kg_venue_error=None,
        kg_topics_ok=False,
        kg_topics_error=None,
        kg_chunks_ok=True,  # still wrote chunks
        kg_chunks_error=None,
        kg_entities_ok=False,  # entities layer off in this corpus
        kg_entities_error=None,
        n_entity_mentions=0,
    )
    assert r.work is None
    assert r.kg_citations_ok is False
    assert r.kg_chunks_ok is True
    assert r.kg_entities_ok is False
    assert r.n_entity_mentions == 0


# ---- ingest_document config-driven gating (mocked) ----
#
# These tests verify only the dispatch decision in Stage 6 of
# `ingest_document` - which KG writes fire based on `config.layers` and
# whether OpenAlex resolved. They mock every external service so they
# stay fast unit tests; integration is covered by `smoke_pipeline.py`.

_DUMMY_PATH = Path("dummy.pdf")
_DUMMY_WORK: dict = {"id": "https://openalex.org/W1"}


def _make_mock_kg() -> MagicMock:
    """KG-client mock where every write reports success (return True).

    `spec=Neo4jClient` auto-creates AsyncMock attributes for every
    `async def` method on the real client, so awaiting them inside
    `asyncio.gather` works without per-test setup.
    """
    mock = MagicMock(spec=Neo4jClient)
    for write in (
        "write_citations",
        "write_authorships",
        "write_venue",
        "write_topics",
        "write_chunks",
        "write_entities",
    ):
        setattr(mock, write, AsyncMock(return_value=True))
    # Default the preserve-labels lookup to "no existing doc" so tests
    # written before this hook don't accidentally trip the preserve
    # branch. Preserve-specific tests override per-call.
    mock.get_focal_labels_by_doc_id = AsyncMock(return_value=(None, None))
    return mock


# Patch order note: decorators apply outside-in, so the first arg to the
# test function is the INNERMOST (last-applied) decorator. embed_texts is
# stubbed to None so the LanceDB write branch is skipped entirely - these
# tests are about KG gating, LanceDB is orthogonal.
@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_skips_openalex_writes_when_layer_off(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """openalex_papers=False -> L1-L4 KG writes skipped even when
    OpenAlex resolved successfully."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = _DUMMY_WORK  # OpenAlex did resolve
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    # L1-L4 writes skipped despite work being resolved and sub_label=Paper.
    mock_kg.write_citations.assert_not_called()
    mock_kg.write_authorships.assert_not_called()
    mock_kg.write_venue.assert_not_called()
    mock_kg.write_topics.assert_not_called()
    # L5 still ran (chunks layer on).
    mock_kg.write_chunks.assert_called_once()
    # Always-on cleanup still ran.
    mock_kg.delete_doc.assert_called_once()
    mock_kg.delete_chunks_by_doc_id.assert_called_once()
    mock_kg.ensure_constraints.assert_called_once()


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_skips_chunk_writes_when_layer_off(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """chunks=False -> L5 write skipped; L1-L4 still run when OpenAlex
    resolves."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = _DUMMY_WORK
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=False),
    )
    await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    # L1-L4 ran.
    mock_kg.write_citations.assert_called_once()
    mock_kg.write_authorships.assert_called_once()
    mock_kg.write_venue.assert_called_once()
    mock_kg.write_topics.assert_called_once()
    # L5 skipped.
    mock_kg.write_chunks.assert_not_called()
    # Always-on cleanup still ran (clears any stale chunks from previous runs).
    mock_kg.delete_chunks_by_doc_id.assert_called_once()


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_runs_all_kg_writes_when_both_layers_on(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Sanity: both layers on + OpenAlex resolved + sub_label=Paper -> every KG write fires."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = _DUMMY_WORK
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.write_citations.assert_called_once()
    mock_kg.write_authorships.assert_called_once()
    mock_kg.write_venue.assert_called_once()
    mock_kg.write_topics.assert_called_once()
    mock_kg.write_chunks.assert_called_once()


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_skips_openalex_writes_when_layer_on_but_work_none(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """openalex_papers=True but OpenAlex didn't resolve -> L1-L4 still
    skipped. Preserves the pre-config behaviour (`work is not None` gate
    remains the second condition)."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = None  # OpenAlex failed
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.write_citations.assert_not_called()
    mock_kg.write_authorships.assert_not_called()
    mock_kg.write_venue.assert_not_called()
    mock_kg.write_topics.assert_not_called()
    # L5 still ran - chunks layer doesn't depend on `work`.
    mock_kg.write_chunks.assert_called_once()


# ---- ingest_document new gates: sub_label gating + input validation ----


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_skips_openalex_when_sub_label_is_not_paper(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Even with openalex_papers=True, a Note-typed (non-Paper) file
    short-circuits the whole metadata step: `resolve_metadata` is NOT
    called (the Paper gate skips DOI extraction + the OpenAlex call), and
    no L1-L4 KG writes happen."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = _DUMMY_WORK
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper", "Note"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    await ingest_document(_DUMMY_PATH, config, "Document", "Note")

    mock_kg.write_citations.assert_not_called()
    mock_kg.write_authorships.assert_not_called()
    mock_kg.write_venue.assert_not_called()
    mock_kg.write_topics.assert_not_called()
    mock_kg.write_chunks.assert_called_once()
    # The Paper gate short-circuits the OpenAlex call itself for a non-Paper.
    mock_resolve.assert_not_called()


async def test_ingest_document_rejects_invalid_main_label():
    config = CorpusConfig(allowed_types=["Paper"])
    with pytest.raises(ValueError, match="main_label"):
        await ingest_document(_DUMMY_PATH, config, "NotAThing", "Paper")


async def test_ingest_document_rejects_sub_label_not_in_allowed_types():
    config = CorpusConfig(allowed_types=["Note"])  # Paper not allowed here
    with pytest.raises(ValueError, match="allowed_types"):
        await ingest_document(_DUMMY_PATH, config, "Document", "Paper")


async def test_ingest_document_rejects_sub_label_wrong_family():
    """A :Document-family sub_label paired with `main_label='Artifact'`
    is rejected at validation time."""
    config = CorpusConfig(allowed_types=["Paper"])
    with pytest.raises(ValueError, match="belongs under"):
        await ingest_document(_DUMMY_PATH, config, "Artifact", "Paper")


async def test_ingest_document_rejects_unsupported_extension(tmp_path: Path):
    # .parquet is the canonical unsupported example - deferred from
    # Phase 4 launch (see roadmap). If/when a Parquet parser ships,
    # swap this for another extension that's still unsupported.
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"PAR1")  # parquet magic, not a real file
    config = CorpusConfig(allowed_types=["Dataset"])
    with pytest.raises(ValueError, match="No parser available"):
        await ingest_document(parquet_path, config, "Artifact", "Dataset")


# ---- ingest_document preserve_existing_labels ----
#
# When the doc's focal node already exists in Neo4j AND
# `preserve_existing_labels=True` (default), the passed main/sub_label
# get swapped for the stored ones before write_chunks fires. When the
# focal doesn't exist, or preserve is False, the passed labels are used.


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_preserve_uses_stored_labels_when_focal_exists(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Existing focal (:Document:Note) + call passes (Document, Paper)
    with preserve=True → write_chunks fires with (Document, Note)."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    # Focal already exists as :Document:Note.
    mock_kg.get_focal_labels_by_doc_id = AsyncMock(
        return_value=("Document", "Note"),
    )
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper", "Note"],
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    await ingest_document(
        _DUMMY_PATH,
        config,
        "Document",
        "Paper",
        preserve_existing_labels=True,
    )

    # Preserve branch swapped Paper -> Note before write_chunks.
    mock_kg.write_chunks.assert_called_once()
    call_args = mock_kg.write_chunks.call_args
    assert call_args.args[2] == "Document"
    assert call_args.args[3] == "Note"


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_preserve_false_forces_passed_labels(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Existing focal (:Document:Note) + preserve=False → write_chunks
    fires with the passed (Document, Paper). Overwrite path."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    # Focal already exists as :Document:Note — should be ignored.
    mock_kg.get_focal_labels_by_doc_id = AsyncMock(
        return_value=("Document", "Note"),
    )
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper", "Note"],
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    await ingest_document(
        _DUMMY_PATH,
        config,
        "Document",
        "Paper",
        preserve_existing_labels=False,
    )

    mock_kg.write_chunks.assert_called_once()
    call_args = mock_kg.write_chunks.call_args
    assert call_args.args[2] == "Document"
    assert call_args.args[3] == "Paper"
    # Preserve was skipped — the lookup should not have been called.
    mock_kg.get_focal_labels_by_doc_id.assert_not_called()


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
async def test_ingest_document_preserve_uses_passed_when_focal_missing(
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """First ingest: focal doesn't exist → preserve is a no-op → the
    passed (Document, Paper) reach write_chunks."""
    mock_parse.return_value = [_chunk(0, "hello")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_kg.get_focal_labels_by_doc_id = AsyncMock(
        return_value=(None, None),
    )
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    await ingest_document(
        _DUMMY_PATH,
        config,
        "Document",
        "Paper",
        preserve_existing_labels=True,
    )

    mock_kg.write_chunks.assert_called_once()
    call_args = mock_kg.write_chunks.call_args
    assert call_args.args[2] == "Document"
    assert call_args.args[3] == "Paper"


# ---- ingest_document L6 (entities) gating ----


def _config_with_entities(
    entity_types: list[str] | None = None,
) -> CorpusConfig:
    """Build a CorpusConfig with the entities layer on."""
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=False, chunks=True, entities=True),
        entities=EntityConfig(extractor="llm", entity_types=entity_types or []),
    )


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_runs_l6_when_entities_layer_on(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """layers.entities=true + chunks layer write OK -> get_extractor + per-chunk
    extract + write_entities all fire."""
    mock_parse.return_value = [_chunk(0, "BRCA1 is in here.")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = [Mention(raw_text="BRCA1", entity_type="GENE")]

    config = _config_with_entities(["GENE"])
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    # The priority-ordered union is invoked once per chunk with the
    # ordered extractor list, the shared entity_types + mode, and the
    # LLM model/temperature forwarded as llm_kwargs (applied only to the
    # "llm" adapter inside extract_union).
    mock_extract_union.assert_called_once_with(
        "BRCA1 is in here.",
        ["llm"],
        ["GENE"],
        entity_types_mode="replace",
        llm_kwargs={
            "model": "claude-haiku-4-5-20251001",
            "temperature": 0.0,
        },
    )
    # write_entities called with one (chunk_id, mentions) tuple.
    mock_kg.write_entities.assert_called_once()
    call_args = mock_kg.write_entities.call_args.args
    assert call_args[0] == "doc-abc"
    chunk_mentions = call_args[1]
    assert len(chunk_mentions) == 1
    chunk_id, mentions = chunk_mentions[0]
    assert chunk_id == "doc-abc#0"
    assert mentions == [Mention(raw_text="BRCA1", entity_type="GENE")]
    # delete_entities runs unconditionally.
    mock_kg.delete_entities_by_doc_id.assert_called_once()
    # IngestResult counts mentions.
    assert result.kg_entities_ok is True
    assert result.n_entity_mentions == 1


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_skips_l6_when_entities_layer_off(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Default config (entities=false) -> no get_extractor, no
    write_entities. delete_entities still fires unconditionally."""
    mock_parse.return_value = [_chunk(0, "BRCA1 is in here.")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),  # entities defaults to False
    )
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_extract_union.assert_not_called()
    mock_kg.write_entities.assert_not_called()
    # Always-on cleanup still ran.
    mock_kg.delete_entities_by_doc_id.assert_called_once()
    assert result.kg_entities_ok is False
    assert result.n_entity_mentions == 0


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_skips_l6_when_chunks_write_fails(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """entities=true but write_chunks raised -> :MENTIONS edges would
    have no :Chunk anchors. Orchestrator records the failure on
    `kg_chunks_error` and skips extraction entirely."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    # Simulate chunk write failure via the typed-errors contract.
    mock_kg.write_chunks.side_effect = RuntimeError("boom")
    mock_get_kg.return_value = mock_kg

    config = _config_with_entities(["GENE"])
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_extract_union.assert_not_called()
    mock_kg.write_entities.assert_not_called()
    assert result.kg_entities_ok is False
    assert result.kg_chunks_ok is False
    assert result.kg_chunks_error is not None
    assert "boom" in result.kg_chunks_error.message


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l6_extractor_exception_skips_only_failing_chunk(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Per-chunk try/except: one chunk failing extraction logs + yields
    empty mentions for that chunk, the rest still process."""
    mock_parse.return_value = [
        _chunk(0, "first chunk"),
        _chunk(1, "second chunk"),
    ]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    # One chunk's union raises; the other returns a mention.
    mock_extract_union.side_effect = [
        RuntimeError("backend died"),
        [Mention(raw_text="TP53", entity_type="GENE")],
    ]

    config = _config_with_entities(["GENE"])
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    # Both chunks attempted; per-chunk failure didn't propagate.
    assert mock_extract_union.call_count == 2
    mock_kg.write_entities.assert_called_once()
    chunk_mentions = mock_kg.write_entities.call_args.args[1]
    assert len(chunk_mentions) == 2
    # Failed chunk -> empty list; successful chunk -> one Mention.
    assert chunk_mentions[0] == ("doc-abc#0", [])
    assert chunk_mentions[1] == (
        "doc-abc#1",
        [Mention(raw_text="TP53", entity_type="GENE")],
    )
    assert result.n_entity_mentions == 1


# Closed-vocabulary integration test removed 2026-06-23 when
# SciSpaCy (the only closed-vocab adapter) was demoted. All current
# adapters (LLM, GLiNER, GLiNER-BioMed, HunFlair2) declare
# KNOWN_LABELS=None so the validator is a no-op for them. The
# generic closed-vocab logic is still covered by dispatcher tests
# in test_entity_extractors_dispatcher.py.


def test_corpus_config_rejects_entities_on_without_chunks():
    """The model validator catches the dependency violation at config
    load time, not in the pipeline."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="layers.chunks"):
        CorpusConfig(
            allowed_types=["Paper"],
            layers=LayerFlags(chunks=False, entities=True),
            entities=EntityConfig(extractor="llm"),
        )


# ---- ingest_document L7 (ontology) gating ----


def _config_with_ontology_mesh(matching: str = "exact") -> CorpusConfig:
    """Build a CorpusConfig with the MeSH ontology layer on."""
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, ontology_mesh=True),
        entities=EntityConfig(extractor="llm", entity_types=["DISEASE"]),
        ontology={"mesh": OntologyConfig(matching=matching)},  # type: ignore[arg-type]
    )


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_skips_l7_when_no_ontology_layer_enabled(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Default config has all ontology_* flags off -> ensure_ontology_imported
    and link_entities_to_ontology are never called."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = []

    config = _config_with_entities(["GENE"])  # entities on, no ontology
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.ensure_ontology_imported.assert_not_called()
    mock_kg.link_entities_to_ontology.assert_not_called()
    assert result.kg_ontology_results == {}


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l7_skipped_when_entities_failed(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """L7 layer enabled but L6 entities-write raises -> orchestrator
    records the failure on `kg_entities_error` and L7 does NOT run
    (no entities to link)."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_kg.write_entities.side_effect = RuntimeError("boom")  # L6 write fails
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = []

    config = _config_with_ontology_mesh()
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.ensure_ontology_imported.assert_not_called()
    mock_kg.link_entities_to_ontology.assert_not_called()
    assert result.kg_ontology_results == {}


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l7_first_import_runs_global_link(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """First-time import (was_already_imported=False) -> linking runs
    with doc_id=None (global pass) so existing entities also get linked."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    # First time: not yet imported (returns False = "wasn't already
    # imported"). Failures raise; this success case returns False
    # under the typed-errors contract.
    mock_kg.ensure_ontology_imported.return_value = False
    mock_kg.link_entities_to_ontology.return_value = 7
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = [Mention(raw_text="x", entity_type="DISEASE")]

    config = _config_with_ontology_mesh()
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.ensure_ontology_imported.assert_called_once_with(
        "mesh",
        xrefs_mode="none",
    )
    # First-time import -> doc_id=None for global linking.
    mock_kg.link_entities_to_ontology.assert_called_once_with("mesh", "exact", doc_id=None)
    assert result.kg_ontology_results == {
        "mesh": {"imported": True, "import_ok": True, "n_links": 7}
    }


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l7_subsequent_ingest_links_only_this_doc(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Already-imported ontology (was_already_imported=True) -> linking
    runs with doc_id=THIS_DOC, only this doc's entities get linked."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_kg.ensure_ontology_imported.return_value = True
    mock_kg.link_entities_to_ontology.return_value = 3
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = [Mention(raw_text="x", entity_type="DISEASE")]

    config = _config_with_ontology_mesh(matching="fuzzy")
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.link_entities_to_ontology.assert_called_once_with("mesh", "fuzzy", doc_id="doc-abc")
    assert result.kg_ontology_results == {
        "mesh": {"imported": False, "import_ok": True, "n_links": 3}
    }


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l7_import_failure_skips_linking_returns_status(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """If ensure_ontology_imported raises (network down, parse error),
    the linking step is skipped but the result still records the
    import attempt as failed for diagnostics."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_kg.ensure_ontology_imported.side_effect = RuntimeError("network down")
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = [Mention(raw_text="x", entity_type="DISEASE")]

    config = _config_with_ontology_mesh()
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    mock_kg.link_entities_to_ontology.assert_not_called()
    assert result.kg_ontology_results == {
        "mesh": {"imported": True, "import_ok": False, "n_links": 0}
    }


@patch(
    "knowledge_agent.ingestion.pipeline.compute_doc_id",
    return_value="doc-abc",
)
@patch("knowledge_agent.ingestion.pipeline.parse_document")
@patch("knowledge_agent.ingestion.pipeline.resolve_metadata", new_callable=AsyncMock)
@patch(
    "knowledge_agent.ingestion.pipeline.collect_doi_candidates",
    return_value=[],
)
@patch(
    "knowledge_agent.ingestion.pipeline.embed_chunks",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("knowledge_agent.ingestion.pipeline.get_kg_client")
@patch("knowledge_agent.ingestion.pipeline.extract_union", new_callable=AsyncMock)
async def test_ingest_document_l7_runs_each_enabled_ontology_independently(
    mock_extract_union,
    mock_get_kg,
    _mock_embed,
    _mock_extract_doi,
    mock_resolve,
    mock_parse,
    _mock_doc_id,
):
    """Two ontology layers enabled -> ensure_ontology_imported and
    link_entities_to_ontology are called once per ontology, with the
    right per-ontology matching strategy from corpus.toml."""
    mock_parse.return_value = [_chunk(0, "text")]
    mock_resolve.return_value = None
    mock_kg = _make_mock_kg()
    mock_kg.ensure_ontology_imported.return_value = (True, True)
    mock_kg.link_entities_to_ontology.return_value = 1
    mock_get_kg.return_value = mock_kg

    mock_extract_union.return_value = [Mention(raw_text="x", entity_type="DISEASE")]

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            ontology_go=True,
        ),
        entities=EntityConfig(extractor="llm", entity_types=["DISEASE"]),
        ontology={
            "mesh": OntologyConfig(matching="exact"),
            "go": OntologyConfig(matching="fuzzy"),
        },
    )
    result = await ingest_document(_DUMMY_PATH, config, "Document", "Paper")

    # Both ontologies got an ensure_imported call.
    ensure_calls = mock_kg.ensure_ontology_imported.call_args_list
    ensure_names = {call.args[0] for call in ensure_calls}
    assert ensure_names == {"mesh", "go"}
    # Both ontologies got a link call, each with its OWN matching strategy.
    link_calls = mock_kg.link_entities_to_ontology.call_args_list
    by_name = {call.args[0]: call for call in link_calls}
    assert by_name["mesh"].args == ("mesh", "exact")
    assert by_name["go"].args == ("go", "fuzzy")
    # Result captures both.
    assert set(result.kg_ontology_results) == {"mesh", "go"}


# ---- L7 xrefs_mode plumbing through ingest + backfill ----


def _config_with_xrefs(
    xrefs_mode: str,
    ontology_names: tuple[str, ...] = ("mesh",),
) -> CorpusConfig:
    """Build a CorpusConfig with `layers.xrefs` set to the given mode
    and the listed ontology layers enabled."""
    flags_kwargs = {
        "chunks": True,
        "entities": True,
        "xrefs": xrefs_mode,
    }
    ontology_cfg = {}
    for n in ontology_names:
        flags_kwargs[f"ontology_{n}"] = True
        ontology_cfg[n] = OntologyConfig(matching="exact")
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(**flags_kwargs),
        entities=EntityConfig(extractor="llm"),
        ontology=ontology_cfg,
    )


async def test_backfill_ontology_threads_xrefs_use_through_ensure_imported():
    """`xrefs="use"` on the config flows through `ensure_ontology_imported`
    as the `xrefs_mode` kwarg — that's how the helper writes resolved
    edges at import time."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=(False, True))
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=0)
    config = _config_with_xrefs("use", ("mesh",))

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        await backfill_ontology("doc-x", config)

    kg_mock.ensure_ontology_imported.assert_called_once_with(
        "mesh",
        xrefs_mode="use",
    )


async def test_backfill_ontology_threads_xrefs_collect_only_mode():
    """`xrefs="collect_only"` flows verbatim. Default behaviour for
    users who want dangling_xrefs stored without writing edges yet."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=(False, True))
    config = _config_with_xrefs("collect_only", ("mesh",))

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        await backfill_ontology("doc-x", config)

    kg_mock.ensure_ontology_imported.assert_called_once_with(
        "mesh",
        xrefs_mode="collect_only",
    )


async def test_backfill_ontology_default_xrefs_mode_is_none():
    """When the config omits `xrefs`, the default `"none"` mode reaches
    the helper — preserves pre-L7-xrefs behaviour."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.ensure_ontology_imported = AsyncMock(return_value=(False, True))
    config = _config_with_ontologies("mesh")  # no xrefs flag set -> default "none"

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        await backfill_ontology("doc-x", config)

    kg_mock.ensure_ontology_imported.assert_called_once_with(
        "mesh",
        xrefs_mode="none",
    )


# ---- L10 backfill_cross_doc_xrefs ----


def _cross_doc_xrefs_enabled_config(
    threshold: int = 2,
) -> CorpusConfig:
    """Build a CorpusConfig with the full L10 dependency chain on:
    entities + xrefs="use" + cross_doc_xrefs + at least one ontology.

    Optionally overrides the threshold by passing it in the
    [cross_doc_xrefs] settings section.
    """
    from knowledge_agent.kg.corpus_config import (
        CrossDocXrefsConfig,
    )

    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            xrefs="use",
            cross_doc_xrefs=True,
        ),
        entities=EntityConfig(extractor="llm"),
        ontology={"mesh": OntologyConfig(matching="exact")},
        cross_doc_xrefs=CrossDocXrefsConfig(threshold=threshold),
    )


async def test_backfill_cross_doc_xrefs_no_op_when_layer_disabled():
    """L10 layer off -> immediate return, no client calls."""
    kg_mock = MagicMock(spec=Neo4jClient)
    config = _entities_enabled_config()  # cross_doc_xrefs is off

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc_xrefs("doc-x", config)

    assert result == {"cross_doc_xrefs_ok": False, "n_edges": 0}
    kg_mock.recompute_cross_doc_xrefs_edges.assert_not_called()


async def test_backfill_cross_doc_xrefs_happy_path_returns_count():
    """recompute returns int -> ok=True, n=that int. Threshold from
    the [cross_doc_xrefs] config block flows through as the positional
    arg to the delegate."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_xrefs_edges = AsyncMock(return_value=7)
    config = _cross_doc_xrefs_enabled_config(threshold=3)

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc_xrefs("doc-1", config)

    assert result == {"cross_doc_xrefs_ok": True, "n_edges": 7}
    kg_mock.recompute_cross_doc_xrefs_edges.assert_called_once_with(
        "doc-1",
        3,
    )


async def test_backfill_cross_doc_xrefs_zero_edges_is_success():
    """No other doc met threshold -> ok=True, n=0 (distinct from
    cross_doc_xrefs_ok=False which means Cypher failure)."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_xrefs_edges = AsyncMock(return_value=0)
    config = _cross_doc_xrefs_enabled_config()

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc_xrefs("doc-1", config)

    assert result == {"cross_doc_xrefs_ok": True, "n_edges": 0}


async def test_backfill_cross_doc_xrefs_exception_means_failure():
    """recompute raising -> ok=False, n=0 (orchestrator catches)."""
    kg_mock = MagicMock(spec=Neo4jClient)
    kg_mock.recompute_cross_doc_xrefs_edges = AsyncMock(side_effect=RuntimeError("boom"))
    config = _cross_doc_xrefs_enabled_config()

    with patch(
        "knowledge_agent.ingestion.pipeline.get_kg_client",
        return_value=kg_mock,
    ):
        result = await backfill_cross_doc_xrefs("doc-1", config)

    assert result == {"cross_doc_xrefs_ok": False, "n_edges": 0}


# ---- IngestResult L10 fields ----


def test_ingest_result_l10_fields_default_to_safe_values():
    """`kg_cross_doc_xrefs_ok=False` + `n_cross_doc_xrefs_edges_written=0`
    by default — same safe-defaults pattern as L9's fields."""
    result = IngestResult(
        doc_id="d",
        path=Path("/tmp/x"),
        n_chunks=0,
        metadata_status="baseline",
        work=None,
        embed_ok=False,
        embed_error=None,
        lancedb_ok=False,
        lancedb_error=None,
        kg_citations_ok=False,
        kg_citations_error=None,
        kg_authorships_ok=False,
        kg_authorships_error=None,
        kg_venue_ok=False,
        kg_venue_error=None,
        kg_topics_ok=False,
        kg_topics_error=None,
        kg_chunks_ok=False,
        kg_chunks_error=None,
        kg_entities_ok=False,
        kg_entities_error=None,
        n_entity_mentions=0,
    )
    assert result.kg_cross_doc_xrefs_ok is False
    assert result.n_cross_doc_xrefs_edges_written == 0
