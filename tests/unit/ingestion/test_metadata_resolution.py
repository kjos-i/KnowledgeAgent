"""Unit tests for `ingestion.metadata_resolution` — the OpenAlex resolution
state machine + the work→columns mapping + author-display, all with mocks
(no network, no DB).

Covers the behaviours that integration tests can't cheaply exercise: the
`manual` protection short-circuit, the `enriched` status transition, the
Paper-gated KG L1–L4 rewrite, and the "unresolved → state unchanged" paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from knowledge_agent.ingestion import metadata_resolution as MR

_MOD = "knowledge_agent.ingestion.metadata_resolution"


def _search_client(rows: list[dict]) -> AsyncMock:
    client = AsyncMock()
    client.get_chunks_by_doc_id = AsyncMock(return_value=rows)
    client.update_doc_metadata = AsyncMock()
    return client


# ---- _doc_metadata_fields_from_work (pure mapping) ----


def test_fields_from_work_none_is_all_none():
    fields = MR._doc_metadata_fields_from_work(None)
    assert set(fields) == {
        "title",
        "year",
        "doi",
        "openalex_id",
        "venue",
        "source_url",
        "authors_display",
        "language",
    }
    assert all(v is None for v in fields.values())


def test_fields_from_work_maps_openalex_shape():
    work = {
        "title": "A Title",
        "publication_year": 2021,
        "doi": "https://doi.org/10.1/x",
        "id": "https://openalex.org/W123",
        "primary_location": {
            "source": {"display_name": "Nature"},
            "landing_page_url": "http://example.org/paper",
        },
        "authorships": [{"author": {"display_name": "Smith"}}],
        "language": "en",
    }
    f = MR._doc_metadata_fields_from_work(work)
    assert f["title"] == "A Title"
    assert f["year"] == 2021
    assert f["venue"] == "Nature"
    assert f["source_url"] == "http://example.org/paper"
    assert f["authors_display"] == "Smith"
    assert f["language"] == "en"
    assert f["doi"] and "10.1/x" in f["doi"]  # normalised by _clean_doi_for_storage
    assert f["openalex_id"] == "W123"  # normalised by _extract_openalex_id


def test_fields_from_work_title_falls_back_to_display_name():
    f = MR._doc_metadata_fields_from_work({"display_name": "Fallback Title"})
    assert f["title"] == "Fallback Title"


# ---- _build_authors_display ----


def test_authors_display_empty_or_nameless_is_none():
    assert MR._build_authors_display([]) is None
    assert MR._build_authors_display([{"author": {}}]) is None


def test_authors_display_joins_up_to_cap():
    authors = [{"author": {"display_name": n}} for n in ("A", "B")]
    assert MR._build_authors_display(authors) == "A, B"


def test_authors_display_adds_et_al_past_cap():
    authors = [{"author": {"display_name": n}} for n in ("A", "B", "C", "D")]
    assert MR._build_authors_display(authors) == "A, B, C, et al."


# ---- resolve_openalex: state machine ----


async def test_resolve_openalex_skips_manual_when_flag_set():
    """skip_manual=True + stored status=manual → protected no-op."""
    rows = [{"sub_label": "Note", "doi": "10.1/x", "metadata_status": "manual"}]
    with patch(f"{_MOD}.get_search_client", return_value=_search_client(rows)):
        result = await MR.resolve_openalex("doc1", skip_manual=True)
    assert result["skipped"] is True
    assert result["work_resolved"] is False


async def test_resolve_openalex_enriches_on_stored_doi_hit():
    rows = [{"sub_label": "Note", "doi": "10.1/x", "metadata_status": "pending"}]
    sc = _search_client(rows)
    with (
        patch(f"{_MOD}.get_search_client", return_value=sc),
        patch(f"{_MOD}.resolve_doi", new=AsyncMock(return_value={"title": "T"})),
    ):
        result = await MR.resolve_openalex("doc1")
    assert result["work_resolved"] is True
    assert result["new_status"] == "enriched"
    assert result["metadata_patched"] is True
    sc.update_doc_metadata.assert_awaited_once()


async def test_resolve_openalex_no_work_leaves_state_unchanged():
    rows = [{"sub_label": "Note", "doi": "10.1/x", "chunk_index": 0, "text": "t"}]
    sc = _search_client(rows)
    with (
        patch(f"{_MOD}.get_search_client", return_value=sc),
        patch(f"{_MOD}.resolve_doi", new=AsyncMock(return_value=None)),
        patch(f"{_MOD}.resolve_metadata", new=AsyncMock(return_value=None)),
    ):
        result = await MR.resolve_openalex("doc1")
    assert result["work_resolved"] is False
    assert result["new_status"] is None
    sc.update_doc_metadata.assert_not_awaited()


# ---- lookup_known_doi ----


async def test_lookup_known_doi_empty_is_noop():
    result = await MR.lookup_known_doi("doc1", "")
    assert result["work_resolved"] is False
    assert result["new_status"] is None


async def test_lookup_known_doi_unresolved_leaves_state_unchanged():
    sc = _search_client([{"sub_label": "Note"}])
    with (
        patch(f"{_MOD}.get_search_client", return_value=sc),
        patch(f"{_MOD}.resolve_doi", new=AsyncMock(return_value=None)),
    ):
        result = await MR.lookup_known_doi("doc1", "10.1/x")
    assert result["work_resolved"] is False
    sc.update_doc_metadata.assert_not_awaited()


# ---- _apply_resolved_work: Paper-gated KG L1–L4 ----


async def test_apply_resolved_work_paper_writes_kg_l1_l4():
    sc = _search_client([])
    kg = AsyncMock()
    with (
        patch(f"{_MOD}.get_search_client", return_value=sc),
        patch(f"{_MOD}.get_kg_client", return_value=kg),
    ):
        result = await MR._apply_resolved_work("doc1", {"title": "T"}, MR.PAPER_LABEL)
    assert result["kg_l1_l4_ok"] is True
    assert result["new_status"] == "enriched"
    kg.write_citations.assert_awaited_once()
    kg.write_topics.assert_awaited_once()


async def test_apply_resolved_work_non_paper_skips_kg():
    sc = _search_client([])
    kg = AsyncMock()
    with (
        patch(f"{_MOD}.get_search_client", return_value=sc),
        patch(f"{_MOD}.get_kg_client", return_value=kg),
    ):
        result = await MR._apply_resolved_work("doc1", {"title": "T"}, "Note")
    assert result["kg_l1_l4_ok"] is False  # L1–L4 doesn't apply to non-Paper
    assert result["metadata_patched"] is True
    kg.write_citations.assert_not_awaited()
