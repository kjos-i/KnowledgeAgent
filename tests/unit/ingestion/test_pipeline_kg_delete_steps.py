"""C7: the per-doc KG delete order is a SINGLE source shared by delete_doc and
re-ingest's pre-wipe (`_kg_wipe`). This pins the load-bearing order — L8 triples
MUST be wiped before the L6 entity GC (the GC's plain DELETE errors on an
:Entity still joined by a surviving L8 edge) — so a regression can't
re-introduce that bug in one path but not the other.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from knowledge_agent.ingestion.pipeline import _doc_kg_delete_steps


def test_doc_kg_delete_steps_preserves_load_bearing_order():
    labels = [label for label, _method in _doc_kg_delete_steps(MagicMock())]
    assert labels == [
        "kg.delete_doc",
        "kg.delete_chunks",
        "kg.delete_triples",  # L8 — MUST precede the entity GC below
        "kg.delete_entities",  # L6
    ]
