"""Per-corpus configuration loaded from `corpus.toml`.

Each corpus folder gets its own `corpus.toml` holding the choices that
vary per corpus (which knowledge-graph layers are active, sub-label
filter, extractor models, and the embedder). Shared infrastructure
settings (Neo4j URI, API keys) stay in `config.py` + `.env`. The
embedder (provider/model/dims) is per-corpus and lives here because
LanceDB pins the vector dimension at ingest — a global embedder would
silently mismatch the moment a second corpus uses a different one.
`apply_corpus_embedding_to_env` (below) bridges the active corpus's
embedder into the environment so `get_settings()` — and thus ingest +
query embedding — resolve to THIS corpus's embedder.

Principle (parallel to `config.py` + `.env`):
  - This module defines the SHAPE (Pydantic models + loader).
  - `corpus.toml` in each corpus folder holds the VALUES.

The future GUI writes `corpus.toml` directly when a user creates or
edits a corpus; nothing else writes it. This loader is read-only and
uses `tomlkit` to parse - reading via tomlkit keeps the door open for
future write paths that need to preserve user comments and formatting.

Why TOML, not YAML:
  - No code-execution risk (TOML is a data-only format; YAML's
    `!!python/object/apply:os.system` family has no analog).
  - No "Norway problem" type coercion (TOML is explicitly typed; YAML
    1.1 would silently coerce unquoted "no" / "off" into False).
  - Comment preservation on round-trip writes via `tomlkit` (PyYAML's
    `yaml.dump` strips comments). Matters when the GUI rewrites the
    file after edits.
  - Same format as `pyproject.toml`, one less syntax for contributors.
"""

import os
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_agent.config import (
    EMBEDDING_DIM_DEFAULTS,
    EMBEDDING_MODEL_DEFAULTS,
    PROVIDER_NODE_DEFAULTS,
    EmbeddingProvider,
)
from knowledge_agent.kg.schema import ALL_SUB_LABELS


class LayerFlags(BaseModel):
    """Per-layer on/off toggles. One field per knowledge-graph layer.

    Adding a new layer means: one new bool field here + one gated call
    in `ingestion/pipeline.py` + one conditional section in
    `kg/schema_as_prompt.py`. A new field's default should be False so
    existing corpora don't silently start writing the new layer when
    the project is upgraded.
    """

    model_config = ConfigDict(extra="forbid")

    openalex_papers: bool = Field(
        default=False,
        description=(
            "L1-L4 bundled: citation graph + author graph + venue + "
            "topics, all derived from a single OpenAlex API lookup per "
            "document. Turn on for scientific paper corpora; off for "
            "non-paper corpora (legal, finance, etc.) where OpenAlex "
            "returns nothing useful."
        ),
    )
    chunks: bool = Field(
        default=True,
        description=(
            "L5: per-chunk `:Chunk` nodes joinable with LanceDB via the "
            "shared `chunk_id`. Required for any corpus where the agent "
            "retrieves and cites specific passages - effectively always "
            "on for retrieval use cases. Default True for that reason."
        ),
    )
    entities: bool = Field(
        default=False,
        description=(
            "L6: chunk-level entity extraction. Writes `:Entity` nodes "
            "and `:MENTIONS` edges from `:Chunk`. Requires an "
            "`[entities]` section in `corpus.toml` declaring the "
            "`extractor`. Independent of `:Document` sub-label - runs "
            "on every chunk regardless of `main_label`/`sub_label`. "
            "Canonical-linking layers (L7: MeSH, GO, ...) get their "
            "own flag family (`ontology_*`)."
        ),
    )
    triples: bool = Field(
        default=False,
        description=(
            "L8: typed entity-to-entity relations extracted by an LLM "
            "from chunk text. Writes one edge per chunk assertion "
            "between :Entity nodes, using one of 15 fixed predicate "
            "edge types (INHIBITS, ACTIVATES, BINDS_TO, ...). Requires "
            "`entities=true` - the LLM is constrained to the L6 "
            "entity vocabulary mined from each chunk. Cost: one "
            "Haiku call per chunk (~$0.05-0.10 per 200-chunk paper). "
            "Per-corpus settings live in `config.py` (model + "
            "temperature) for now; per-corpus overrides via a "
            "[triples] section are deferred until needed."
        ),
    )
    cross_doc: bool = Field(
        default=False,
        description=(
            "L9: cross-document synthesis - materialised `:RELATED_TO` "
            "edges between two documents (or artifacts) that share at "
            "least N L6 `:Entity` nodes (default N=2). Undirected; "
            "edge carries the shared entity keys + count + timestamp. "
            "Requires `entities=true` - the shared signal is L6 "
            "entities (NOT canonicals, so this layer is unaffected by "
            "ontology import/delete). No LLM cost: pure Cypher pass. "
            "Recomputed per-doc on ingest / backfill. Threshold "
            "configurable via the `[cross_doc]` section."
        ),
    )
    xrefs: Literal["none", "collect_only", "use"] = Field(
        default="none",
        description=(
            "L7 cross-ontology xref edges - 3-state flag. Source "
            "ontologies declare equivalent / closely-related concepts "
            "in OTHER ontologies (OBO `xref:` lines, oboInOwl "
            "`hasDbXref`, SKOS `closeMatch` / `exactMatch`). This "
            "flag controls whether those declarations are materialised "
            "into the graph at import time.\n"
            '- `"none"` (default): don\'t extract xrefs at all. No '
            "  property storage, no edges. Re-enabling later requires "
            "  force re-importing affected ontologies.\n"
            '- `"collect_only"`: extract xrefs, store the verbatim '
            "  list as a `dangling_xrefs` property on each source term, "
            "  but DON'T write resolved edges yet. Lets the user flip "
            '  to `"use"` later via the fast `backfill_xrefs` '
            "  bulk_op (no re-import needed).\n"
            '- `"use"`: extract xrefs, store the property, AND write '
            "  resolved `:<X>_XREF` edges immediately at import for "
            "  every xref whose target already exists. Targets that "
            "  don't exist yet (e.g. an ontology not imported) stay "
            "  in `dangling_xrefs` until backfill picks them up.\n"
            "Storage cost: ~30-40 MB across the biomedical ontology "
            "set (negligible on a multi-GB graph). Parse cost: ~5%."
        ),
    )
    cross_doc_xrefs: bool = Field(
        default=False,
        description=(
            "L10: concept-level cross-document synthesis - materialised "
            "`:RELATED_BY_XREF` edges between two documents (or "
            "artifacts) that share at least N canonical concepts via "
            "xref equivalence (default N=2). Parallel to L9 but uses "
            "the xref graph: two docs share a concept when their "
            "entities canonicalise to `:OntologyTerm` nodes that are "
            "EITHER the same node OR connected by a `:<X>_XREF` edge. "
            "Surfaces cross-ontology equivalence that L9's raw-entity "
            "signal can't reach (same drug under MeSH and ChEBI, same "
            "disease under MONDO and DOID). Requires `entities=true` "
            'AND `xrefs="use"` (xref edges must exist). Threshold '
            "configurable via the `[cross_doc_xrefs]` section. "
            "Recomputed per-doc on ingest / backfill. No LLM cost: "
            "pure Cypher pass. Heavy on a large corpus (~minutes to "
            "rebuild graph-wide); per-doc cost stays small."
        ),
    )
    ontology_mesh: bool = Field(
        default=False,
        description=(
            "L7: import MeSH (Medical Subject Headings) into the KG as "
            "`:MeSHTerm` nodes + `:MESH_BROADER` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical MeSH IDs "
            "via `:CANONICAL_TO` edges. Requires `entities=true` (the "
            "linking step needs entities to link to). One-time import "
            "happens automatically on first ingestion after enabling. "
            "MeSH is the broad biomedical vocabulary (diseases, "
            "chemicals, anatomy); enable for biomedical corpora."
        ),
    )
    ontology_go: bool = Field(
        default=False,
        description=(
            "L7: import GO (Gene Ontology) into the KG as `:GOTerm` "
            "nodes + `:GO_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical GO IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Covers "
            "biological processes, molecular functions, and cellular "
            "components; enable for cell biology / biochemistry "
            "corpora."
        ),
    )
    ontology_hpo: bool = Field(
        default=False,
        description=(
            "L7: import HPO (Human Phenotype Ontology) into the KG as "
            "`:HPOTerm` nodes + `:HPO_IS_A` hierarchy edges, then link "
            "existing `:Entity` nodes to their canonical HPO IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Covers "
            "human phenotypic abnormalities (clinical features, signs, "
            "symptoms, lab abnormalities, behavioural traits); enable "
            "for biomedical corpora where phenotype-level granularity "
            "matters (e.g. rare disease, clinical genetics)."
        ),
    )
    ontology_uberon: bool = Field(
        default=False,
        description=(
            "L7: import UBERON (Uber Anatomy Ontology) into the KG as "
            "`:UBERONTerm` nodes + `:UBERON_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical UBERON "
            "IDs via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Multi-species anatomy (tissues, organs, body parts); "
            "enable for biomedical / comparative biology corpora that "
            "reference anatomical structures."
        ),
    )
    ontology_mondo: bool = Field(
        default=False,
        description=(
            "L7: import MONDO (Mondo Disease Ontology) into the KG as "
            "`:MONDOTerm` nodes + `:MONDO_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical MONDO "
            "IDs via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Integrates DOID, OMIM, Orphanet, EFO disease branch, "
            "NCIT diseases, ICD-11; enable for biomedical corpora "
            "that reference diseases."
        ),
    )
    ontology_chebi: bool = Field(
        default=False,
        description=(
            "L7: import ChEBI (Chemical Entities of Biological "
            "Interest, LITE variant) into the KG as `:ChEBITerm` "
            "nodes + `:CHEBI_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical ChEBI IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Small "
            "molecules with biological relevance (drugs, metabolites, "
            "signaling molecules, nutrients); enable for biomedical / "
            "biochemistry / pharmacology corpora that reference "
            "chemicals."
        ),
    )
    ontology_eco: bool = Field(
        default=False,
        description=(
            "L7: import ECO (Evidence & Conclusion Ontology) into the "
            "KG as `:ECOTerm` nodes + `:ECO_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical ECO IDs "
            "via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Evidence codes used to qualify annotations (experimental "
            "evidence, sequence orthology, author statement); enable "
            "for corpora that reference how annotations were derived."
        ),
    )
    ontology_so: bool = Field(
        default=False,
        description=(
            "L7: import SO (Sequence Ontology) into the KG as `:SOTerm` "
            "nodes + `:SO_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical SO IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Sequence "
            "features (gene, promoter, intron, CDS) and variant types "
            "(SNV, indel, structural variant); enable for genomics "
            "corpora."
        ),
    )
    ontology_pr: bool = Field(
        default=False,
        description=(
            "L7: import PR (Protein Ontology) into the KG as `:PRTerm` "
            "nodes + `:PR_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical PR IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Protein "
            "classes, isoforms, complexes, and ortholog relationships "
            "(adds granularity UniProt's flat model collapses); enable "
            "for proteins corpora."
        ),
    )
    ontology_cl: bool = Field(
        default=False,
        description=(
            "L7: import CL (Cell Ontology) into the KG as `:CLTerm` "
            "nodes + `:CL_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical CL IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Cell "
            "types across organisms (neurons, lymphocytes, epithelial "
            "cells); enable for cell biology corpora."
        ),
    )
    ontology_po: bool = Field(
        default=False,
        description=(
            "L7: import PO (Plant Ontology) into the KG as `:POTerm` "
            "nodes + `:PO_IS_A` hierarchy edges, then link existing "
            "`:Entity` nodes to their canonical PO IDs via "
            "`:CANONICAL_TO` edges. Requires `entities=true`. Plant "
            "anatomy + developmental stages (root, leaf, flower, "
            "anthesis); fills UBERON's vertebrate-skewed gap. Enable "
            "for plant biology corpora."
        ),
    )
    ontology_foodon: bool = Field(
        default=False,
        description=(
            "L7: import FOODON (Food Ontology) into the KG as "
            "`:FOODONTerm` nodes + `:FOODON_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical FOODON "
            "IDs via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Foods, food products, dietary components, processing "
            "methods; enable for nutrition corpora."
        ),
    )
    ontology_envo: bool = Field(
        default=False,
        description=(
            "L7: import ENVO (Environment Ontology) into the KG as "
            "`:ENVOTerm` nodes + `:ENVO_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical ENVO IDs "
            "via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Biomes, environmental features, materials, exposures; "
            "enable for environment / ecology corpora."
        ),
    )
    ontology_ncbitaxon: bool = Field(
        default=False,
        description=(
            "L7: import NCBI Taxonomy into the KG as `:NCBITaxonTerm` "
            "nodes + `:NCBITAXON_IS_A` hierarchy edges, then link "
            "existing `:Entity` nodes to their canonical NCBITaxon IDs "
            "via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Biological organism taxonomy (species, genus, family, ...). "
            "WARNING: ~2.74M classes, ~440 MB source - by far the "
            "largest ontology on the menu. Budget several GB of RAM "
            "for the first-time import; see ncbitaxon_writes "
            "module docstring."
        ),
    )
    ontology_obi: bool = Field(
        default=False,
        description=(
            "L7: import OBI (Ontology for Biomedical Investigations) "
            "into the KG as `:OBITerm` nodes + `:OBI_IS_A` hierarchy "
            "edges, then link existing `:Entity` nodes to their "
            "canonical OBI IDs via `:CANONICAL_TO` edges. Requires "
            "`entities=true`. Experimental methods, assay protocols, "
            "study designs, laboratory equipment; enable for "
            "experimental-methods corpora. OWL/RDF source parsed via "
            "rdflib (not pronto) so the synonym surface is preserved."
        ),
    )
    ontology_efo: bool = Field(
        default=False,
        description=(
            "L7: import EFO (Experimental Factor Ontology) into the "
            "KG as `:EFOTerm` nodes + `:EFO_IS_A` hierarchy edges, "
            "then link existing `:Entity` nodes to their canonical "
            "EFO IDs via `:CANONICAL_TO` edges. Requires "
            "`entities=true`. EBI's integrative experimental-factors "
            "vocabulary (diseases, cell types, cell lines, traits, "
            "measurement methods); enable for biomedical corpora "
            "working with EBI study data. OWL/RDF source parsed via "
            "rdflib."
        ),
    )
    ontology_dron: bool = Field(
        default=False,
        description=(
            "L7: import DRON (Drug Ontology) into the KG as "
            "`:DRONTerm` nodes + `:DRON_IS_A` hierarchy edges, then "
            "link existing `:Entity` nodes to their canonical DRON "
            "IDs via `:CANONICAL_TO` edges. Requires `entities=true`. "
            "Drug products, ingredients, dose forms, packaging - "
            "built on RxNorm with realist upper-level structure. "
            "WARNING: ~700K classes, ~220 MB OWL source - second-"
            "largest ontology on the menu after NCBITaxon. Budget "
            "several GB of RAM for the first-time import; see "
            "dron_writes module docstring."
        ),
    )
    ontology_fibo: bool = Field(
        default=False,
        description=(
            "L7: import FIBO (Financial Industry Business Ontology) "
            "into the KG as `:FIBOTerm` nodes + `:FIBO_IS_A` hierarchy "
            "edges, then link existing `:Entity` nodes to their "
            "canonical FIBO IDs via `:CANONICAL_TO` edges. Requires "
            "`entities=true`. Financial instruments, transactions, "
            "parties, agreements, corporations, market infrastructure. "
            "Distributes modularly as ~70 .rdf files on GitHub; the "
            "FIBO module's walker fetches each on first install and "
            "caches them locally (subsequent imports re-use the cache). "
            "First-time install takes 1-2 min of network."
        ),
    )


class EntityConfig(BaseModel):
    """L6 entity-extraction settings.

    Required when `layers.entities=true`; ignored otherwise.
    `CorpusConfig`'s model validator enforces that pairing.

    Validation of each extractor name against the dispatcher's known
    adapter set happens at extraction time (not here) - that check
    lives with the dispatcher to avoid a kg/ -> entity_extractors/
    import dependency.

    Multi-extractor (priority-ordered union): `extractors` is an
    ORDERED list. Index 0 is the base/primary (owns overlaps); each
    later extractor contributes only spans no higher-priority extractor
    already found. This is fragmentation-free by construction (one
    owner per span). Back-compat: a legacy singular `extractor = "llm"`
    is auto-migrated to `extractors = ["llm"]` by the validator below,
    and the `extractor` property returns the primary for old read
    sites.
    """

    model_config = ConfigDict(extra="forbid")

    extractors: list[str] = Field(
        min_length=1,
        description=(
            "Ordered list of adapter names from `entity_extractors/` "
            "(e.g. ['hunflair2', 'llm']). Order = priority: index 0 is "
            "the base (owns overlapping spans); each later extractor "
            "adds only spans the earlier ones didn't find. A single-"
            "element list (e.g. ['llm']) is the common case. The "
            "dispatcher validates each name at extraction time and "
            "raises if no adapter matches. When a NER and the LLM are "
            "combined, the recommended order is NER first (its high-"
            "precision domain types own overlaps) then LLM (fills the "
            "gaps its fixed vocab can't reach)."
        ),
    )
    entity_types: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form list of entity-type labels the corpus cares about "
            "(e.g. ['GENE', 'DISEASE', 'CHEMICAL']). Empty list = use "
            "each adapter's default behaviour: the LLM adapter is "
            "prompted to categorise entities freely; zero-shot NER "
            "adapters return their DEFAULT_LABELS; HunFlair2 returns its "
            "fixed model label set regardless. NOT validated against "
            "any project-wide enum - entity types are per-corpus user "
            "vocabulary, distinct from the fixed sub-label list. Shared "
            "across all selected extractors; each interprets it per its "
            "nature (see `entity_types_mode`)."
        ),
    )
    entity_types_mode: Literal["replace", "add"] = Field(
        default="replace",
        description=(
            "How a non-empty `entity_types` list interacts with each "
            "extractor's own default labels. 'replace' (default): the "
            "typed list is used ALONE - the adapter's defaults are not "
            "included. 'add': the typed list is merged WITH each "
            "adapter's DEFAULT_LABELS (so you can extend the defaults "
            "with a custom label without re-listing them all). Only "
            "meaningful for adapters that HAVE defaults (the zero-shot "
            "GLiNER adapters); the LLM has no fixed defaults (so 'add' "
            "≈ 'replace' for it) and HunFlair2 ignores `entity_types` "
            "entirely. Ignored when `entity_types` is empty."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_singular_extractor(cls, data: Any) -> Any:
        """Back-compat: map a legacy singular `extractor = "x"` onto the
        new `extractors = ["x"]` list. Existing `corpus.toml` files +
        older `EntityConfig(extractor="llm")` call sites keep working.
        When both keys are present, `extractors` wins and the singular
        is dropped (so `extra="forbid"` doesn't reject it)."""
        if isinstance(data, dict) and "extractor" in data:
            legacy = data.get("extractor")
            data = {k: v for k, v in data.items() if k != "extractor"}
            if "extractors" not in data:
                if isinstance(legacy, str):
                    data["extractors"] = [legacy]
                elif legacy is not None:
                    data["extractors"] = list(legacy)
        return data

    @property
    def extractor(self) -> str:
        """Primary (highest-priority) extractor name. Back-compat
        accessor for the pre-list single-extractor field; equals
        `extractors[0]`."""
        return self.extractors[0]


class OntologyConfig(BaseModel):
    """L7 per-ontology settings.

    One instance per enabled ontology (e.g. one for MeSH, one for GO).
    All fields have sensible defaults so users can enable an ontology
    layer without writing an `[ontology.<name>]` section if they're
    happy with the defaults. The `CorpusConfig` model validator auto-
    populates default `OntologyConfig` for any enabled `ontology_*`
    layer that doesn't have an explicit section.
    """

    model_config = ConfigDict(extra="forbid")

    matching: Literal["exact", "fuzzy"] = Field(
        default="exact",
        description=(
            "Linking strategy for matching `:Entity` nodes to ontology "
            "terms. 'exact' (default) only links entities whose `key` "
            "exactly matches a term's label or synonym (lowercased). "
            "'fuzzy' falls through to a more permissive match when exact "
            "fails: it tries a small fixed set of string variants "
            "(trailing-'s' singular/plural flip, hyphen/space swap, and "
            "the two combined), not edit-distance or spell-correction. It "
            "catches simple morphological variation at some risk of "
            "false positives. Switch to 'fuzzy' if 'exact' leaves too "
            "many entities unlinked, then re-run the linking pass."
        ),
    )


class CrossDocConfig(BaseModel):
    """L9 cross-document synthesis settings (`[cross_doc]` section).

    Optional; the `CorpusConfig` model validator auto-populates a
    default instance whenever `layers.cross_doc=true` and the section
    was omitted. Single tunable: the minimum number of shared L6
    entity keys required to materialise a `:RELATED_TO` edge between
    two docs.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum number of distinct shared L6 `:Entity` keys "
            "required to materialise a `:RELATED_TO` edge between two "
            "docs. Default 2: single-shared-entity = noise floor "
            "(common generic terms like 'patient' or 'study'); two "
            "distinct shared entities = meaningful signal. Bump higher "
            "(3-5) to tighten the relationship graph at the cost of "
            "recall; lower to 1 only for very small corpora where "
            "every overlap matters. Must be ≥ 1."
        ),
    )


class CrossDocXrefsConfig(BaseModel):
    """L10 cross-document synthesis-via-xref settings
    (`[cross_doc_xrefs]` section).

    Optional; the `CorpusConfig` model validator auto-populates a
    default instance whenever `layers.cross_doc_xrefs=true` and the
    section was omitted. Single tunable: the minimum number of shared
    canonical concepts (via xref equivalence) required to materialise
    a `:RELATED_BY_XREF` edge between two docs.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum number of distinct shared canonical concepts "
            "(counted on the focal doc's side, joined via xref "
            "equivalence) required to materialise a `:RELATED_BY_XREF` "
            "edge between two docs. Default 2: single-shared-concept "
            "= noise floor (high-level shared term like 'Disease' "
            "via MeSH-MONDO xref); two distinct shared concepts = "
            "meaningful topical overlap. Tune the same way as L9's "
            "threshold. Must be ≥ 1."
        ),
    )


class CorpusConfig(BaseModel):
    """Settings that vary per corpus, loaded from `corpus.toml`.

    Each corpus folder holds its own `corpus.toml`. This dataclass is
    the parsed result; the rest of the code reads `config.layers.<name>`
    to decide which write paths run during ingest and which schema
    sections appear in the LLM prompts at query time.
    """

    model_config = ConfigDict(extra="forbid")

    frozen: bool = Field(
        default=False,
        description=(
            "When True, the corpus's ingestion recipe is locked in the GUI: "
            "the embedder and the L6–L10 graph-layer settings (entities, "
            "ontologies, triples, cross-doc, cross-doc xrefs) become read-only "
            "so they can't change accidentally between ingests. Per-batch "
            "settings (labels, paper resolution, chunking/parse/figure options) "
            "stay editable. Ingests and bulk operations still run under the "
            "frozen recipe. A GUI-only guard — the backend never enforces it."
        ),
    )

    layers: LayerFlags = Field(
        default_factory=LayerFlags,
        description="Per-layer on/off toggles. See LayerFlags.",
    )
    allowed_types: list[str] = Field(
        default_factory=lambda: list(ALL_SUB_LABELS),
        description=(
            "Sub-labels this corpus accepts at ingest time. Each entry "
            "must be a known sub-label name from `schema.ALL_SUB_LABELS` "
            "(case-sensitive, e.g. 'Paper', 'Note', 'Dataset'). Default "
            "= every known sub-label — a fresh corpus can be tagged with "
            "any of them. Narrow the list to restrict a corpus (e.g. "
            "`['Paper']` for a paper-only corpus rejects everything else "
            "at ingest). Explicit `[]` = no sub-label allowed (files "
            "still ingest but only get a top-level `:Document` / "
            "`:Artifact` label). `ingest_document` validates the "
            "caller's `sub_label` argument against this list."
        ),
    )
    enable_pdf_ocr: bool = Field(
        default=False,
        description=(
            "OCR for PDF inputs. Default off — research papers are "
            "usually born-digital (text already embedded), so OCR adds "
            "latency for no gain. Turn on for scanned / image-only PDFs. "
            "Per-corpus (the previous global `Settings.enable_pdf_ocr` "
            "is no longer read at ingest time; each corpus captures its "
            "own value at creation)."
        ),
    )
    enable_image_ocr: bool = Field(
        default=True,
        description=(
            "OCR for standalone image-format inputs (PNG / JPG / TIFF as "
            "the whole document). Default on — without OCR, image-only "
            "inputs produce no searchable text. Per-corpus."
        ),
    )
    chunker_strategy: Literal["hybrid", "hierarchical"] = Field(
        default="hybrid",
        description=(
            "Which docling chunker to use. 'hybrid' (default) is "
            "structure- AND token-aware: respects headings / paragraphs "
            "/ tables AND splits anything over `chunk_max_tokens`. "
            "'hierarchical' is structure-only with no token cap — chunks "
            "align to document sections exactly but can be arbitrarily "
            "large. Per-corpus. Changing this on a corpus with existing "
            "chunks yields inconsistent chunk shapes across ingests "
            "until those documents are re-ingested (backfill does not "
            "re-chunk; it re-reads the existing chunk text)."
        ),
    )
    chunk_max_tokens: int = Field(
        default=512,
        ge=64,
        description=(
            "Max tokens per chunk (HybridChunker only — ignored when "
            '`chunker_strategy="hierarchical"`). Higher = broader '
            "context per chunk, fewer chunks total, more storage per "
            "chunk. Per-corpus. Changing this on a corpus with existing "
            "chunks yields inconsistent chunk sizes across ingests "
            "until those documents are re-ingested (backfill does not "
            "re-chunk; it re-reads the existing chunk text)."
        ),
    )
    merge_peers: bool = Field(
        default=True,
        description=(
            "HybridChunker knob (ignored when "
            '`chunker_strategy="hierarchical"`). When true, adjacent '
            "chunks in the same section that both fit under "
            "`chunk_max_tokens` are greedy-merged into one — fewer, "
            "larger chunks. False leaves each structure-aligned unit "
            "as its own chunk. Per-corpus."
        ),
    )
    images_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Docling PDF-render scale factor applied to PDF and "
            "standalone IMAGE inputs. Higher = more detail per rendered "
            "page (better OCR fidelity on small text; slower to render). "
            "Per-corpus. Meaningful only when OCR is on."
        ),
    )
    extract_figures: bool = Field(
        default=False,
        description=(
            "Multimodal: extract figure images from PDF / Word / PPT "
            "sources during parsing. Each picture is saved as "
            "`<corpus_root>/figures/<doc_id>/<i>.png` for downstream "
            "embedding + display. Standalone image files (PNG / JPG / "
            "TIFF) always reference the original file in place, "
            "regardless of this toggle. Default off — text-only corpora "
            "skip the extra parse work. Per-corpus."
        ),
    )
    embed_images: bool = Field(
        default=False,
        description=(
            "Multimodal: produce figure chunks (content_type='figure') "
            "at chunking time and send image + caption to the "
            "multimodal embedder. Requires a multimodal-capable "
            "embedding provider (Voyage today). When off, only text "
            "chunks are produced even if `extract_figures=true` (the "
            "PNGs get saved but not embedded). Default off. Per-corpus."
        ),
    )
    min_figure_bytes: int = Field(
        default=2048,
        ge=0,
        description=(
            "Multimodal: minimum size in bytes for a saved figure PNG "
            "to be kept. Files smaller than this are deleted right "
            "after being written and no figure chunk is emitted. "
            "Docling can flag tiny decorative pictures (page banners, "
            "logos, single-glyph icons) around 400-1000 B; the 2 KB "
            "default filters those without touching real diagrams. "
            "Set to 0 to disable the filter entirely and keep every "
            "picture Docling emits. Applies only when "
            "`extract_figures=true`. Per-corpus."
        ),
    )
    optimize_indexes_per_ingest: bool = Field(
        default=True,
        description=(
            "When true, `ingest_document` calls "
            "`LanceClient.ensure_indexes()` after each successful write "
            "so LanceDB vector + FTS indexes stay current. Turn off for "
            "bulk ingest sessions when you'd rather defer index rebuild "
            "to a single optimize at the end. Per-corpus."
        ),
    )
    entity_extractor_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["entity_extractor"],
        description=(
            "Model used by the LLM entity-extractor adapter (L6). "
            "Haiku is cheap + fast — one call per chunk, short input, "
            "straightforward span extraction (~$0.05 per 200-chunk "
            "paper at current Haiku pricing). Consulted whenever "
            "'llm' is among `entities.extractors` (any priority "
            "position), not only as the primary. Per-corpus — a biomedical "
            "corpus wanting Sonnet-grade extraction picks it here "
            "without disturbing other corpora."
        ),
    )
    entity_extractor_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the LLM entity-extractor adapter. 0.0 = "
            "deterministic — structured Pydantic output (ExtractedMentions) "
            "works best at temperature 0. Consulted whenever 'llm' is "
            "among `entities.extractors` (any priority position). Per-corpus."
        ),
    )
    triples_extractor_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["triples_extractor"],
        description=(
            "Model used by the L8 LLM triples-extractor. Haiku by "
            "default (~$0.05 per 200-chunk paper). Bumping to a "
            "stronger model helps when the L6 entity vocabulary is "
            "dense and the LLM needs to disambiguate many candidate "
            "relations. Consulted only when `layers.triples=true`. "
            "Per-corpus — a biomedical corpus wanting Sonnet-grade "
            "extraction picks it here without disturbing other corpora."
        ),
    )
    triples_extractor_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the L8 LLM triples-extractor. 0.0 = "
            "deterministic — the constrained predicate vocabulary + "
            "structured Pydantic output (ExtractedTriples) benefit "
            "from stable output. Consulted only when "
            "`layers.triples=true`. Per-corpus."
        ),
    )
    embedding_provider: EmbeddingProvider = Field(
        default="voyage",
        description=(
            "Embedding provider for THIS corpus. The embedder is physically "
            "corpus-bound — LanceDB pins the vector dimension at table "
            "creation — so it lives here, per-corpus, NOT in global settings "
            "(same rationale as the extractor models above: baked into the "
            "persisted store). Voyage is the default (multimodal-capable). "
            "Changing it on a corpus that already holds chunks is destructive "
            "(needs a re-embed); the dimension guard in "
            "`embedder_lifecycle.switch_embedder_plan` surfaces that. Default "
            "mirrors `Settings.embedding_provider`."
        ),
    )
    embedding_model: str = Field(
        default=EMBEDDING_MODEL_DEFAULTS["voyage"],
        description=(
            "Embedding model for THIS corpus's provider. Must stay consistent "
            "with the chunks already in LanceDB — the vector dimension is "
            "pinned at table creation. Default mirrors `Settings.embedding_model`."
        ),
    )
    embedding_dims: int = Field(
        default=EMBEDDING_DIM_DEFAULTS["voyage"],
        ge=1,
        description=(
            "Vector dimension produced by this corpus's embedding model "
            "(voyage-multimodal-3 = 1024, OpenAI = 1536, Google = 768, HF "
            "varies). Fixed by the model; LanceDB pins it at table creation, "
            "so it must match the existing chunks. Default mirrors "
            "`Settings.embedding_dims`."
        ),
    )
    entities: EntityConfig | None = Field(
        default=None,
        description=(
            "L6 entity-extraction settings. Required when "
            "`layers.entities=true`; ignored otherwise (the model "
            "validator below enforces that pairing). When the "
            "`entities` layer flag is off, this section can be omitted "
            "entirely."
        ),
    )
    ontology: dict[str, OntologyConfig] = Field(
        default_factory=dict,
        description=(
            "L7 per-ontology settings, keyed by ontology name (e.g. "
            "'mesh', 'go'). Optional - any enabled `ontology_*` layer "
            "without an explicit `[ontology.<name>]` section gets a "
            "default `OntologyConfig()`. Only the ontologies whose "
            "corresponding `layers.ontology_*` flag is true are read at "
            "runtime; configs for disabled ontologies are accepted but "
            "ignored (lets users prepare configs before flipping the "
            "flag)."
        ),
    )
    cross_doc: CrossDocConfig | None = Field(
        default=None,
        description=(
            "L9 cross-document settings. Optional - the model "
            "validator auto-populates `CrossDocConfig()` when "
            "`layers.cross_doc=true` and the section is omitted. "
            "Ignored when the layer flag is off. NOT the same as "
            "`layers.cross_doc` (the on/off bool); this holds the "
            "threshold."
        ),
    )
    cross_doc_xrefs: CrossDocXrefsConfig | None = Field(
        default=None,
        description=(
            "L10 cross-document-xrefs settings. Optional - the model "
            "validator auto-populates `CrossDocXrefsConfig()` when "
            "`layers.cross_doc_xrefs=true` and the section is omitted. "
            "Ignored when the layer flag is off. NOT the same as "
            "`layers.cross_doc_xrefs` (the on/off bool); this holds "
            "the threshold."
        ),
    )

    @field_validator("allowed_types")
    @classmethod
    def _check_allowed_types_are_known(cls, value: list[str]) -> list[str]:
        """Reject typos / unknown sub-labels at config-load time.

        Catches `allowed_types = ["Paer"]` (typo) in the corpus.toml
        before any ingest happens, rather than waiting for a runtime
        ValidationError when a user tries to ingest a `Paer`-typed file.
        """
        unknown = [t for t in value if t not in ALL_SUB_LABELS]
        if unknown:
            raise ValueError(
                f"Unknown sub-label(s) in allowed_types: {unknown}. "
                f"Valid sub-labels are: {sorted(ALL_SUB_LABELS)}."
            )
        return value

    @model_validator(mode="after")
    def _check_entities_section_required_when_layer_on(self):
        """Refuse `layers.entities=true` without a matching `[entities]`
        section. The layer would have nothing to dispatch to otherwise."""
        if self.layers.entities and self.entities is None:
            raise ValueError(
                "layers.entities=true requires an [entities] section "
                "in corpus.toml declaring at least 'extractor'. Either "
                "add the section, or set layers.entities=false to skip "
                "entity extraction."
            )
        return self

    @model_validator(mode="after")
    def _check_entities_requires_chunks(self):
        """`:MENTIONS` edges anchor to `:Chunk` nodes; without chunks
        there is nothing to attach mentions to. Catch this at config
        load rather than after we've already extracted mentions and
        produced dangling `:Entity` nodes that have no provenance edges."""
        if self.layers.entities and not self.layers.chunks:
            raise ValueError(
                "layers.entities=true requires layers.chunks=true - "
                "the :MENTIONS edges written by the entity layer anchor "
                "to :Chunk nodes from the chunk layer, so chunks must "
                "be written first. Turn chunks on, or turn entities off."
            )
        return self

    @model_validator(mode="after")
    def _check_triples_requires_entities(self):
        """L8 LLM extractor is constrained to the L6 entity vocabulary.
        Without entities there is no vocabulary to constrain to, and
        all triples would be dropped at the extractor's post-validation
        step. Catch at config load rather than after spending LLM
        budget producing triples we'd throw away."""
        if self.layers.triples and not self.layers.entities:
            raise ValueError(
                "layers.triples=true requires layers.entities=true - "
                "the L8 extractor uses each chunk's L6 entity "
                "vocabulary as the constrained subject/object set. "
                "Turn entities on, or turn triples off."
            )
        return self

    @model_validator(mode="after")
    def _check_cross_doc_requires_entities(self):
        """L9 :RELATED_TO edges materialise overlap between two docs'
        L6 entity sets. Without entities every overlap is the empty
        set and no edge would ever be created - the layer is silently
        useless. Catch at config load."""
        if self.layers.cross_doc and not self.layers.entities:
            raise ValueError(
                "layers.cross_doc=true requires layers.entities=true - "
                "L9 materialises edges between docs that share L6 "
                "entities. Without entities, every overlap is empty "
                "and no edges would ever be written. Turn entities "
                "on, or turn cross_doc off."
            )
        return self

    @model_validator(mode="after")
    def _check_cross_doc_xrefs_requires_entities_and_xrefs_use(self):
        """L10 :RELATED_BY_XREF edges walk
        `Doc -> Chunk -> :Entity -> :CANONICAL_TO -> :OntologyTerm`
        and join across the xref graph. Both halves must exist:
        `entities=true` for the :Entity / :CANONICAL_TO path, and
        `xrefs="use"` for the :<X>_XREF edges that drive equivalence.
        Without entities the path is empty; without xref edges the
        join collapses to identity-only (which would still produce
        L9-like results but waste the user's stated intent). Catch
        at config load so the layer flag's promise matches reality."""
        if self.layers.cross_doc_xrefs:
            if not self.layers.entities:
                raise ValueError(
                    "layers.cross_doc_xrefs=true requires "
                    "layers.entities=true - L10 walks "
                    "Doc->Chunk->Entity->CANONICAL_TO->OntologyTerm, "
                    "which has no Entity nodes to anchor without the "
                    "entities layer. Turn entities on, or turn "
                    "cross_doc_xrefs off."
                )
            if self.layers.xrefs != "use":
                raise ValueError(
                    "layers.cross_doc_xrefs=true requires "
                    'layers.xrefs="use" - L10\'s equivalence join '
                    "relies on the :<X>_XREF edges that the xrefs "
                    "layer writes in 'use' mode. With xrefs=\"none\" "
                    'or "collect_only", no xref edges exist and '
                    "L10 would only catch identity equivalence. Set "
                    'xrefs="use" (running backfill_xrefs after if '
                    "ontologies are already imported), or turn "
                    "cross_doc_xrefs off."
                )
        return self

    @model_validator(mode="after")
    def _check_ontology_requires_entities(self):
        """Every L7 ontology layer needs L6 entities to link to. The
        ontology import itself works without entities (the term nodes
        get written either way), but the linking pass is the point of
        L7 and it needs entities. Catch at load time rather than
        importing a 600 MB ontology only to find there's nothing to
        link it to."""
        enabled = self._enabled_ontology_layers()
        if enabled and not self.layers.entities:
            raise ValueError(
                f"Ontology layers {enabled} require layers.entities=true. "
                f"The :CANONICAL_TO edges from the linking pass connect "
                f":Entity nodes (from L6) to ontology terms; without "
                f"entities, the imported ontology nodes have nothing to "
                f"link to. Turn entities on, or turn the ontology "
                f"layer(s) off."
            )
        return self

    @model_validator(mode="after")
    def _auto_populate_ontology_configs(self):
        """For each enabled `ontology_<name>` layer flag, ensure an
        OntologyConfig exists at `self.ontology[name]`. Lets users omit
        explicit `[ontology.<name>]` sections when defaults are fine.
        Idempotent - existing user-provided configs are preserved."""
        for name in self._enabled_ontology_layers():
            if name not in self.ontology:
                self.ontology[name] = OntologyConfig()
        return self

    @model_validator(mode="after")
    def _auto_populate_cross_doc_configs(self):
        """Auto-populate `cross_doc` / `cross_doc_xrefs` settings
        sections when their layer flag is on but the user omitted the
        `[cross_doc]` / `[cross_doc_xrefs]` block. Mirrors the
        `_auto_populate_ontology_configs` pattern — lets users enable
        a layer without writing its config section if they're happy
        with the defaults. Idempotent — explicit user-provided configs
        are preserved."""
        if self.layers.cross_doc and self.cross_doc is None:
            self.cross_doc = CrossDocConfig()
        if self.layers.cross_doc_xrefs and self.cross_doc_xrefs is None:
            self.cross_doc_xrefs = CrossDocXrefsConfig()
        return self

    def _enabled_ontology_layers(self) -> list[str]:
        """Names of currently-enabled L7 ontology layers (e.g. ['mesh',
        'go']). Reads `ontology_<name>` boolean flags off `self.layers`
        and strips the prefix. Used by the validators above and by
        downstream callers wanting to know which ontology modules to
        dispatch to."""
        layer_dict = self.layers.model_dump()
        return [
            name.removeprefix("ontology_")
            for name, enabled in layer_dict.items()
            if name.startswith("ontology_") and enabled
        ]


def load_corpus_config(path: Path) -> CorpusConfig:
    """Read a `corpus.toml` file and return a validated `CorpusConfig`.

    `path` points at the TOML file itself (not the corpus folder).
    Missing file raises FileNotFoundError; the caller decides whether
    to fall back to defaults (e.g. `CorpusConfig()`) or surface the
    error to the user.

    Parsed with `tomlkit.loads` - TOML has no code-execution surface,
    so this is safe regardless of who wrote the file. Empty TOML files
    yield a default `CorpusConfig`.
    """
    text = path.read_text(encoding="utf-8")
    raw = dict(tomlkit.loads(text))
    return CorpusConfig.model_validate(raw)


def apply_corpus_embedding_to_env(config: CorpusConfig) -> None:
    """Write a corpus's embedder settings into the environment.

    Sets `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMS` so that
    `get_settings()` — and therefore both ingest (`embed_chunks`) and query
    (`embed_texts`) embedding, plus the LanceDB table schema
    (`chunks_schema` reads `embedding_dims`) — resolve to THIS corpus's
    embedder instead of any global default.

    The caller MUST clear cached settings afterwards
    (`config.reset_after_key_change()`); `get_settings` is `lru_cache`d, so
    the new env values only take effect after a clear (same contract as the
    key/connection bridges). Callers: the GUI corpus-switch bridge
    (`gui.config_store.apply_active_corpus_embedding_to_env`) and the
    headless CLI + eval entrypoints, which load a `corpus.toml` and then run
    the embedding factory.

    Only the three resolved fields are written — the factory reads the
    active model from `EMBEDDING_MODEL` for every provider (see
    `embedder_factory`), so no per-provider env var is needed.
    """
    os.environ["EMBEDDING_PROVIDER"] = config.embedding_provider
    os.environ["EMBEDDING_MODEL"] = config.embedding_model
    os.environ["EMBEDDING_DIMS"] = str(config.embedding_dims)


def corpus_folder(corpus_toml_path: Path) -> Path:
    """Return the corpus folder — the directory that owns
    `corpus.toml`, `lancedb/`, and (with multimodal on) `figures/`.

    Convention from Create New Dataset: the user picks one folder to
    hold everything for a corpus. `corpus.toml` sits inside it;
    `lancedb/` is a sub-directory; figures land in `figures/` next to
    LanceDB. This helper returns the folder itself so callers can
    resolve sub-paths (LanceDB, figures, future backups) without
    duplicating the `.parent` derivation.
    """
    return corpus_toml_path.parent


def corpus_figures_dir(corpus_dir: Path, doc_id: str) -> Path:
    """Return the per-doc figures directory beside the LanceDB folder.

    Path is `<corpus folder>/figures/<doc_id>/` — a sibling of
    `<corpus folder>/lancedb/`. Figures are the app's own artefact, not
    LanceDB internals, so they live beside the vector store rather than
    inside its directory.

    The single source of truth for the figures location. Pure path join
    (no mkdir); the ingest pipeline passes `lancedb_path.parent` as the
    corpus folder, and the parser creates the directory when it writes a
    PNG (`<returned dir>/<i>.png`).
    """
    return corpus_dir / "figures" / doc_id
