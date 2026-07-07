"""Shared baseline-vs-draft config diff.

Two views need to show exactly what the user has edited in the config panel
but not yet ingested:

  * the Ingest tab's "changes since last ingest" summary card, and
  * the Select tab's corpus card ("what settings this dataset was ingested
    with, and what's been changed from that since").

Computing the diff in one place keeps the two from ever disagreeing (single
source of truth). ``config_diff`` returns one ``(field_label, baseline_value,
draft_value)`` triple per field whose draft value differs from the saved
baseline, as ready-to-render display strings. The list is empty when the
draft equals the baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.library.corpus_config_editor import _ONTOLOGY_DISPLAY

if TYPE_CHECKING:
    from knowledge_agent.kg.corpus_config import CorpusConfig

# (field_label, baseline_str, draft_str)
ConfigChange = tuple[str, str, str]

# Per-corpus scalar fields compared verbatim (bools rendered on/off).
_SCALAR_FIELDS: tuple[str, ...] = (
    "chunker_strategy",
    "chunk_max_tokens",
    "merge_peers",
    "enable_pdf_ocr",
    "enable_image_ocr",
    "images_scale",
    "optimize_indexes_per_ingest",
    "entity_extractor_model",
    "entity_extractor_temperature",
    "triples_extractor_model",
    "triples_extractor_temperature",
)


def config_diff(baseline: CorpusConfig, draft: CorpusConfig) -> list[ConfigChange]:
    """Every field where ``draft`` differs from ``baseline``, as display strings.

    Mirrors the field set the config editor lets the user toggle. Empty when
    the two configs are equal.
    """
    diffs: list[ConfigChange] = []

    # Layer bool flags.
    for field in (
        "openalex_papers",
        "chunks",
        "entities",
        "triples",
        "cross_doc",
        "cross_doc_xrefs",
    ):
        b = getattr(baseline.layers, field, False)
        d = getattr(draft.layers, field, False)
        if b != d:
            diffs.append((field, _fmt_bool(b), _fmt_bool(d)))

    # xrefs is a 3-state string ("off" / "build" / "use").
    if baseline.layers.xrefs != draft.layers.xrefs:
        diffs.append(("xrefs", baseline.layers.xrefs, draft.layers.xrefs))

    # Per-ontology enable flags.
    for key, display in _ONTOLOGY_DISPLAY.items():
        b = getattr(baseline.layers, f"ontology_{key}", False)
        d = getattr(draft.layers, f"ontology_{key}", False)
        if b != d:
            diffs.append((f"ontology_{key} ({display})", _fmt_bool(b), _fmt_bool(d)))

    # Extractor + entity_types (on the entities subsection).
    base_extractor = baseline.entities.extractor if baseline.entities is not None else "—"
    draft_extractor = draft.entities.extractor if draft.entities is not None else "—"
    if base_extractor != draft_extractor:
        diffs.append(("extractor", base_extractor, draft_extractor))
    base_types = (
        ", ".join(baseline.entities.entity_types) if baseline.entities is not None else "—"
    ) or "(default)"
    draft_types = (
        ", ".join(draft.entities.entity_types) if draft.entities is not None else "—"
    ) or "(default)"
    if base_types != draft_types:
        diffs.append(("entity_types", base_types, draft_types))

    # Per-corpus scalar fields (chunking / OCR / figures / extractor models).
    for name in _SCALAR_FIELDS:
        base_val = getattr(baseline, name)
        draft_val = getattr(draft, name)
        if base_val != draft_val:
            if isinstance(base_val, bool):
                diffs.append((name, _fmt_bool(base_val), _fmt_bool(draft_val)))
            else:
                diffs.append((name, str(base_val), str(draft_val)))

    # Cross-doc thresholds (nested; default 2 when the subsection is absent).
    base_thr = baseline.cross_doc.threshold if baseline.cross_doc is not None else 2
    draft_thr = draft.cross_doc.threshold if draft.cross_doc is not None else 2
    if base_thr != draft_thr:
        diffs.append(("cross_doc.threshold", str(base_thr), str(draft_thr)))
    base_xthr = baseline.cross_doc_xrefs.threshold if baseline.cross_doc_xrefs is not None else 2
    draft_xthr = draft.cross_doc_xrefs.threshold if draft.cross_doc_xrefs is not None else 2
    if base_xthr != draft_xthr:
        diffs.append(("cross_doc_xrefs.threshold", str(base_xthr), str(draft_xthr)))

    return diffs


def _fmt_bool(v: bool) -> str:
    """Render a bool as the on/off tokens the config cards use."""
    return "on" if v else "off"
