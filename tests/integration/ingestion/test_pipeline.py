"""Integration tests for `ingestion/pipeline` — `ingest_document`
end-to-end against the real stores.

This is the heaviest integration tier file: each test exercises
docling parsing + Voyage embeddings + OpenAlex resolution + LanceDB
write + Neo4j L1-L5 write at minimum. To keep cost predictable, all
tests use a corpus config with the higher layers (L6a entities, L7
ontology linking, L8 triples, L9 cross-doc) OFF — those are tested
in isolation by the per-layer integration tests + manually verified
in `scripts/smoke_pipeline.py`.

Cost guard: ~one Voyage embedding call per test + one OpenAlex HTTP
call. ~$0.001 per test run.

Manual interactive counterpart: `scripts/smoke_pipeline.py` (full
stack, all layers on, with MeSH linking).

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from knowledge_agent.ingestion.ids import compute_doc_id
from knowledge_agent.ingestion.pipeline import (
    delete_doc,
    ingest_document,
)
from knowledge_agent.kg.corpus_config import CorpusConfig, LayerFlags

pytestmark = pytest.mark.integration


def _minimal_corpus_config() -> CorpusConfig:
    """Corpus config with L1-L5 + LanceDB on, everything else OFF.

    Keeps the integration tests cheap: no L6a (no LLM-per-chunk
    extraction cost), no L7 (no ~600 MB MeSH download), no L8/L9/L10.
    The per-layer integration tests cover those flags individually.
    """
    return CorpusConfig(
        domain="generic",
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=False,
            triples=False,
            cross_doc=False,
            cross_doc_xrefs=False,
            xrefs="none",
        ),
        allowed_types=["Paper"],
    )


@pytest.fixture
def clean_both_stores(
    kg_client: Any, lance_client: Any, ensure_constraints: None
) -> Any:
    """Wipe both stores before each pipeline test.

    The pipeline writes to both LanceDB and Neo4j, so per-test
    isolation needs both wiped. We can't use `clean_kg` + `clean_lance`
    separately because their teardown order matters (drop lance
    before clear KG is fine; we do KG first then lance for symmetry).
    """
    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    lance_client.drop_chunks_table()
    return None


def test_ingest_document_full_l1_l5_path(
    kg_client: Any, lance_client: Any, clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """A real PDF ingest with L1-L5 enabled produces a successful
    IngestResult with all expected `_ok` flags True + non-zero counts
    on both stores."""
    config = _minimal_corpus_config()
    result = ingest_document(sample_pdf, config, "Document", "Paper")

    # The doc was parsed.
    assert result.n_chunks > 0
    # doc_id is deterministic.
    assert result.doc_id == compute_doc_id(sample_pdf)
    # Metadata resolution worked (the sample PDF has a DOI).
    assert result.metadata_status == "enriched"
    assert result.work is not None
    # Embedding + LanceDB write.
    assert result.embed_ok is True
    assert result.lancedb_ok is True
    # KG L1-L4 writes.
    assert result.kg_citations_ok is True
    assert result.kg_authorships_ok is True
    assert result.kg_venue_ok is True
    assert result.kg_topics_ok is True
    # KG L5 chunks.
    assert result.kg_chunks_ok is True
    # L6a OFF → False but no error.
    assert result.kg_entities_ok is False
    # L8-L10 OFF.
    assert result.kg_triples_ok is False
    assert result.kg_cross_doc_ok is False
    assert result.kg_cross_doc_xrefs_ok is False


def test_ingest_document_writes_chunks_to_both_stores(
    kg_client: Any, lance_client: Any, clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """After ingest, the same n_chunks land in BOTH LanceDB (chunk
    rows) and Neo4j (L5 :Chunk nodes). Catches per-store drift."""
    config = _minimal_corpus_config()
    result = ingest_document(sample_pdf, config, "Document", "Paper")
    doc_id = result.doc_id

    # LanceDB side.
    lance_rows = lance_client.get_chunks_by_doc_id(doc_id)
    assert lance_rows is not None
    assert len(lance_rows) == result.n_chunks

    # Neo4j side.
    with kg_client.driver.session() as session:
        n_kg_chunks = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_kg_chunks == result.n_chunks


def test_ingest_document_is_idempotent_on_doc_id(
    kg_client: Any, lance_client: Any, clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """Re-ingesting the same file does NOT duplicate rows — the per-
    doc delete step at the start of `ingest_document` ensures fresh
    write."""
    config = _minimal_corpus_config()
    first = ingest_document(sample_pdf, config, "Document", "Paper")
    second = ingest_document(sample_pdf, config, "Document", "Paper")
    doc_id = first.doc_id

    assert second.doc_id == doc_id  # same content → same id
    assert second.n_chunks == first.n_chunks

    # LanceDB: row count matches n_chunks (not 2x).
    lance_rows = lance_client.get_chunks_by_doc_id(doc_id)
    assert lance_rows is not None
    assert len(lance_rows) == first.n_chunks

    # Neo4j: chunk count matches (not 2x).
    with kg_client.driver.session() as session:
        n_kg = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_kg == first.n_chunks


def test_delete_doc_wipes_both_stores(
    kg_client: Any, lance_client: Any, clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """The pipeline's top-level `delete_doc` removes the document
    from BOTH stores (LanceDB chunks + Neo4j L1-L5 + orphan GC)."""
    config = _minimal_corpus_config()
    result = ingest_document(sample_pdf, config, "Document", "Paper")
    doc_id = result.doc_id

    # Pre-condition: data present in both stores.
    assert lance_client.get_chunks_by_doc_id(doc_id)
    with kg_client.driver.session() as session:
        n_pre = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_pre == 1

    ok = delete_doc(doc_id)
    assert ok is True

    # Post: both stores empty for this doc.
    lance_rows = lance_client.get_chunks_by_doc_id(doc_id) or []
    assert len(lance_rows) == 0
    with kg_client.driver.session() as session:
        n_post = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_post == 0
