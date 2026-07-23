"""Tests for corpus_config - CorpusConfig + LayerFlags + load_corpus_config.

The model defines the SHAPE of per-corpus settings; `corpus.toml` files
in each corpus folder hold the VALUES. These tests cover defaults,
loader parsing, and Pydantic validation. They do NOT test how
`ingest_document` or the agent CONSUME the loaded config - those tests
live with their respective modules.

Why no "rejects python tag" test (as we had with PyYAML's `safe_load`):
TOML has no code-execution surface - it's a data-only format. There is
no analog to YAML's `!!python/object/apply:os.system` to test against.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from tomlkit.exceptions import ParseError

from knowledge_agent.corpus_config import (
    CorpusConfig,
    CrossDocConfig,
    CrossDocXrefsConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
    apply_corpus_embedding_to_env,
    corpus_figures_dir,
    corpus_folder,
    load_corpus_config,
)

# ---- defaults ----


def test_layer_flags_defaults():
    """openalex_papers safe-default OFF; chunks safe-default ON;
    entities + ontology layers OFF (must opt in)."""
    flags = LayerFlags()
    assert flags.openalex_papers is False
    assert flags.chunks is True
    assert flags.entities is False
    assert flags.ontology_mesh is False
    assert flags.ontology_go is False


def test_corpus_config_defaults():
    """Default config: default layer flags, no entities section, empty
    ontology dict."""
    config = CorpusConfig()
    assert config.layers.openalex_papers is False
    assert config.layers.chunks is True
    assert config.layers.entities is False
    assert config.layers.ontology_mesh is False
    assert config.entities is None
    assert config.ontology == {}


# ---- embedder (per-corpus) ----


def test_corpus_config_embedding_defaults():
    """The embedder is per-corpus; its defaults mirror the Settings global
    default (Voyage multimodal-3, 1024-dim)."""
    config = CorpusConfig()
    assert config.embedding_provider == "voyage"
    assert config.embedding_model == "voyage-multimodal-3"
    assert config.embedding_dims == 1024


def test_load_corpus_config_embedding_section(tmp_path: Path):
    """An explicit embedder in corpus.toml is parsed verbatim."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        'embedding_provider = "openai"\n'
        'embedding_model = "text-embedding-3-large"\n'
        "embedding_dims = 1536\n"
    )
    config = load_corpus_config(toml_path)
    assert config.embedding_provider == "openai"
    assert config.embedding_model == "text-embedding-3-large"
    assert config.embedding_dims == 1536


def test_corpus_config_rejects_unknown_embedding_provider():
    """The provider is a closed Literal — a typo fails at load, not at the
    first embed call."""
    with pytest.raises(ValidationError):
        CorpusConfig(embedding_provider="voyaage")  # type: ignore[arg-type]


def test_apply_corpus_embedding_to_env_sets_the_three_vars(monkeypatch):
    """Writes exactly the three resolved env vars the factory
    (EMBEDDING_PROVIDER/MODEL) + LanceDB schema (EMBEDDING_DIMS) read.
    monkeypatch.setenv records the originals so the direct os.environ writes
    are rolled back after the test."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "OLD")
    monkeypatch.setenv("EMBEDDING_MODEL", "OLD")
    monkeypatch.setenv("EMBEDDING_DIMS", "0")
    config = CorpusConfig(
        embedding_provider="google",
        embedding_model="models/text-embedding-004",
        embedding_dims=768,
    )
    apply_corpus_embedding_to_env(config)
    assert os.environ["EMBEDDING_PROVIDER"] == "google"
    assert os.environ["EMBEDDING_MODEL"] == "models/text-embedding-004"
    assert os.environ["EMBEDDING_DIMS"] == "768"


# ---- load_corpus_config: happy path ----


def test_load_corpus_config_full_toml(tmp_path: Path):
    """All fields specified - parsed verbatim."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nopenalex_papers = true\nchunks = true\n")
    config = load_corpus_config(toml_path)
    assert config.layers.openalex_papers is True
    assert config.layers.chunks is True


def test_load_corpus_config_omitted_fields_use_defaults(tmp_path: Path):
    """Only `allowed_types` set - layers stay at their defaults."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("allowed_types = []\n")
    config = load_corpus_config(toml_path)
    assert config.allowed_types == []
    assert config.layers.openalex_papers is False  # default
    assert config.layers.chunks is True  # default


def test_load_corpus_config_partial_layers(tmp_path: Path):
    """Only one layer set - the other still gets its default."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nopenalex_papers = true\n")
    config = load_corpus_config(toml_path)
    assert config.layers.openalex_papers is True
    assert config.layers.chunks is True  # default


def test_load_corpus_config_empty_toml_returns_defaults(tmp_path: Path):
    """Empty TOML file = no values = all defaults."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("")
    config = load_corpus_config(toml_path)
    assert config.layers.openalex_papers is False
    assert config.layers.chunks is True


def test_load_corpus_config_comment_only_toml_returns_defaults(tmp_path: Path):
    """TOML file with only comments + whitespace = no values = defaults."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("# just a comment, no data\n\n# another comment\n")
    config = load_corpus_config(toml_path)
    assert config.layers.chunks is True


def test_load_corpus_config_layers_off_for_legal_corpus(tmp_path: Path):
    """Realistic non-paper-corpus example: chunks on, openalex off."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nopenalex_papers = false\nchunks = true\n")
    config = load_corpus_config(toml_path)
    assert config.layers.openalex_papers is False
    assert config.layers.chunks is True


# ---- load_corpus_config: error paths ----


def test_load_corpus_config_missing_file_raises(tmp_path: Path):
    """Missing file = caller's decision; raise FileNotFoundError."""
    toml_path = tmp_path / "nonexistent.toml"
    with pytest.raises(FileNotFoundError):
        load_corpus_config(toml_path)


def test_load_corpus_config_unknown_top_level_field_raises(tmp_path: Path):
    """extra='forbid' catches typos at the top level."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "allowed_types = []\nalowed_types = []\n"  # typo of 'allowed_types'
    )
    with pytest.raises(ValidationError):
        load_corpus_config(toml_path)


def test_load_corpus_config_unknown_layer_name_raises(tmp_path: Path):
    """extra='forbid' on LayerFlags catches typos in layer names."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\nopenalex_paper = true\n"  # missing 's'
    )
    with pytest.raises(ValidationError):
        load_corpus_config(toml_path)


def test_load_corpus_config_wrong_type_raises(tmp_path: Path):
    """Non-bool value for a bool field surfaces ValidationError.

    Note: TOML is explicitly typed - `"not_a_bool"` is a string, not a
    bool, and Pydantic flags the mismatch. (In YAML the value would
    parse as a string too, but TOML's explicit typing means a literal
    bool would have to be written as `true`/`false` without quotes -
    no Norway problem.)
    """
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('[layers]\nopenalex_papers = "not_a_bool"\n')
    with pytest.raises(ValidationError):
        load_corpus_config(toml_path)


def test_load_corpus_config_invalid_toml_syntax_raises(tmp_path: Path):
    """Malformed TOML surfaces tomlkit.exceptions.ParseError,
    not a silent default."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("allowed_types = \n")  # missing value
    with pytest.raises(ParseError):
        load_corpus_config(toml_path)


def test_load_corpus_config_top_level_scalar_raises(tmp_path: Path):
    """A bare scalar at TOML root is invalid syntax (TOML requires
    key=value or table headers); parser raises before Pydantic sees it.

    This is stricter than the equivalent YAML behaviour, where a bare
    string parses successfully and Pydantic catches it. Either way the
    caller gets an error, but with TOML it's caught earlier and with a
    syntax-level message rather than a validation-level message.
    """
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("just_a_string\n")
    with pytest.raises(ParseError):
        load_corpus_config(toml_path)


# ---- allowed_types validation ----


def test_corpus_config_allowed_types_defaults_to_all_sub_labels():
    """No declared types -> every known sub-label allowed. A fresh
    corpus can be tagged with any of them; users narrow the list to
    restrict a corpus (e.g. `['Paper']` for paper-only)."""
    from knowledge_agent.kg.schema import ALL_SUB_LABELS

    config = CorpusConfig()
    assert set(config.allowed_types) == set(ALL_SUB_LABELS)


def test_corpus_config_allowed_types_explicit_empty_is_preserved():
    """Explicit empty list means the user opted OUT of sub-label tagging
    — ingest still works (files just get :Document or :Artifact) but
    any sub_label arg gets rejected."""
    config = CorpusConfig(allowed_types=[])
    assert config.allowed_types == []


def test_load_corpus_config_accepts_known_sub_labels(tmp_path: Path):
    """A realistic biomedical-paper corpus declaring three sub-labels."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('allowed_types = ["Paper", "Note", "Dataset"]\n')
    config = load_corpus_config(toml_path)
    assert config.allowed_types == ["Paper", "Note", "Dataset"]


def test_load_corpus_config_rejects_unknown_sub_label(tmp_path: Path):
    """Typo in allowed_types surfaces at config-load time."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('allowed_types = ["Paper", "Paer"]\n')  # 'Paer' typo
    with pytest.raises(ValidationError, match="Paer"):
        load_corpus_config(toml_path)


def test_load_corpus_config_rejects_wrong_case_sub_label(tmp_path: Path):
    """Sub-labels are case-sensitive - 'paper' (lowercase) is invalid
    because the KG label is `:Paper`."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('allowed_types = ["paper"]\n')
    with pytest.raises(ValidationError):
        load_corpus_config(toml_path)


# ---- entities (L6) section ----


def test_entity_config_extractor_required():
    """`extractor` has no default - omitting it raises ValidationError."""
    with pytest.raises(ValidationError):
        EntityConfig()  # type: ignore[call-arg]


def test_entity_config_entity_types_defaults_empty():
    """`entity_types` defaults to []. Adapter-specific behaviour
    (open vocab / full label set) is documented per adapter."""
    config = EntityConfig(extractor="llm")
    assert config.entity_types == []


def test_entity_config_accepts_arbitrary_entity_types():
    """`entity_types` is free-form - any list of strings is accepted at
    config load. NER adapters cross-check against their KNOWN_LABELS
    at extraction time, but corpus_config does NOT validate against any
    project-wide enum (these are per-corpus user vocabulary)."""
    config = EntityConfig(
        extractor="llm",
        entity_types=["GENE", "DISEASE", "MADE_UP_TYPE", "weird_label"],
    )
    assert config.entity_types == [
        "GENE",
        "DISEASE",
        "MADE_UP_TYPE",
        "weird_label",
    ]


def test_entity_config_rejects_unknown_field():
    """`extra='forbid'` catches typos in the [entities] section."""
    with pytest.raises(ValidationError):
        EntityConfig(extractor="llm", entiti_types=["GENE"])  # type: ignore[call-arg]


# ---- multi-extractor contract (priority-ordered union) ----


def test_entity_config_migrates_legacy_singular_extractor():
    """A legacy `extractor='llm'` maps onto `extractors=['llm']` so old
    corpus.toml files + call sites keep working."""
    config = EntityConfig(extractor="llm")
    assert config.extractors == ["llm"]


def test_entity_config_accepts_ordered_extractors_list():
    """`extractors` preserves order (index 0 = base/primary)."""
    config = EntityConfig(extractors=["hunflair2", "llm"])
    assert config.extractors == ["hunflair2", "llm"]


def test_entity_config_extractor_property_returns_primary():
    """The back-compat `extractor` property is `extractors[0]`."""
    config = EntityConfig(extractors=["hunflair2", "llm"])
    assert config.extractor == "hunflair2"


def test_entity_config_both_keys_prefers_extractors():
    """When both the legacy singular and the new list are given, the
    list wins and the singular is dropped (extra='forbid' happy)."""
    config = EntityConfig(extractor="llm", extractors=["gliner"])  # type: ignore[call-arg]
    assert config.extractors == ["gliner"]


def test_entity_config_empty_extractors_rejected():
    """An empty extractor list is invalid - at least one is required."""
    with pytest.raises(ValidationError):
        EntityConfig(extractors=[])


def test_entity_config_missing_extractors_rejected():
    """Neither `extractor` nor `extractors` -> ValidationError."""
    with pytest.raises(ValidationError):
        EntityConfig()  # type: ignore[call-arg]


def test_entity_config_entity_types_mode_defaults_replace():
    """`entity_types_mode` defaults to 'replace' (preserves the pre-
    multi-extractor behaviour where a typed list overrides defaults)."""
    config = EntityConfig(extractors=["llm"])
    assert config.entity_types_mode == "replace"


def test_entity_config_entity_types_mode_accepts_add():
    """'add' is the other valid mode (merge typed list with defaults)."""
    config = EntityConfig(extractors=["gliner"], entity_types_mode="add")
    assert config.entity_types_mode == "add"


def test_entity_config_entity_types_mode_rejects_unknown():
    """Only 'replace' / 'add' are valid modes."""
    with pytest.raises(ValidationError):
        EntityConfig(extractors=["llm"], entity_types_mode="merge")  # type: ignore[arg-type]


def test_entity_config_dump_uses_extractors_not_singular():
    """Serialization emits the new `extractors` field (so saved configs
    persist in the new format), not the legacy `extractor`."""
    config = EntityConfig(extractor="llm")
    dumped = config.model_dump()
    assert "extractors" in dumped
    assert "extractor" not in dumped


def test_load_corpus_config_entities_section_parsed(tmp_path: Path):
    """Happy path: [entities] section parses into EntityConfig."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "entities = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
        'entity_types = ["GENE", "DISEASE"]\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.entities is True
    assert config.entities is not None
    assert config.entities.extractor == "llm"
    assert config.entities.entity_types == ["GENE", "DISEASE"]


def test_load_corpus_config_entities_section_optional_when_layer_off(
    tmp_path: Path,
):
    """layers.entities=false (the default) -> [entities] section can be
    omitted entirely. `config.entities` stays None."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("allowed_types = []\n")
    config = load_corpus_config(toml_path)
    assert config.layers.entities is False
    assert config.entities is None


def test_load_corpus_config_layer_on_without_section_raises(tmp_path: Path):
    """layers.entities=true without an [entities] section -> the layer
    would have nothing to dispatch to. Model validator rejects this at
    load time rather than waiting for a NoneType error in the pipeline."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nentities = true\n")
    with pytest.raises(ValidationError, match="entities"):
        load_corpus_config(toml_path)


def test_load_corpus_config_entities_section_without_layer_on(tmp_path: Path):
    """An [entities] section present but layers.entities=false is fine -
    the section is parsed but ignored at extraction time. This lets a
    user have the section prepared before flipping the flag on."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('[entities]\nextractor = "llm"\n')
    config = load_corpus_config(toml_path)
    assert config.layers.entities is False
    assert config.entities is not None
    assert config.entities.extractor == "llm"


def test_load_corpus_config_entities_rejects_unknown_field(tmp_path: Path):
    """Typos inside the [entities] section surface at config-load time."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        '[layers]\nentities = true\n[entities]\nextractor = "llm"\nextracter = "llm"\n'  # typo
    )
    with pytest.raises(ValidationError):
        load_corpus_config(toml_path)


# ---- L7 ontology layers ----


def test_ontology_config_defaults():
    """OntologyConfig.matching defaults to 'exact'."""
    config = OntologyConfig()
    assert config.matching == "exact"


def test_ontology_config_accepts_fuzzy():
    config = OntologyConfig(matching="fuzzy")
    assert config.matching == "fuzzy"


def test_ontology_config_rejects_unknown_matching_value():
    """`matching` is a Literal of "exact"/"fuzzy" only - other strings
    raise at config load. Surfaces typos like 'exact-match' fast."""
    with pytest.raises(ValidationError):
        OntologyConfig(matching="loose")  # type: ignore[arg-type]


def test_ontology_config_rejects_unknown_field():
    """`extra='forbid'` catches typos in the [ontology.<name>] section."""
    with pytest.raises(ValidationError):
        OntologyConfig(matchng="exact")  # type: ignore[call-arg]


def test_load_corpus_config_ontology_section_parsed(tmp_path: Path):
    """Happy path: [ontology.mesh] populates config.ontology['mesh']."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "ontology_mesh = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
        "[ontology.mesh]\n"
        'matching = "fuzzy"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.ontology_mesh is True
    assert "mesh" in config.ontology
    assert config.ontology["mesh"].matching == "fuzzy"


def test_load_corpus_config_ontology_section_optional_uses_defaults(
    tmp_path: Path,
):
    """`ontology_mesh = true` with NO explicit `[ontology.mesh]` section
    -> auto-populated with default OntologyConfig (matching='exact')."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "ontology_mesh = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert "mesh" in config.ontology
    assert config.ontology["mesh"].matching == "exact"


def test_load_corpus_config_ontology_section_without_layer_on_kept(
    tmp_path: Path,
):
    """User can prepare an `[ontology.mesh]` section before flipping the
    flag - config is parsed and stored, just not active."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text('[ontology.mesh]\nmatching = "fuzzy"\n')
    config = load_corpus_config(toml_path)
    assert config.layers.ontology_mesh is False
    assert "mesh" in config.ontology
    assert config.ontology["mesh"].matching == "fuzzy"


def test_load_corpus_config_ontology_layer_requires_entities(tmp_path: Path):
    """`ontology_mesh=true` without `entities=true` -> the linking pass
    would have nothing to link. Validator rejects at load time."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nchunks = true\nontology_mesh = true\n")
    with pytest.raises(ValidationError, match="entities"):
        load_corpus_config(toml_path)


def test_load_corpus_config_multiple_ontology_layers(tmp_path: Path):
    """Both MeSH and GO can be enabled at once, with per-ontology
    configs side-by-side."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "ontology_mesh = true\n"
        "ontology_go = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
        "[ontology.mesh]\n"
        'matching = "exact"\n'
        "[ontology.go]\n"
        'matching = "fuzzy"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.ontology_mesh is True
    assert config.layers.ontology_go is True
    assert config.ontology["mesh"].matching == "exact"
    assert config.ontology["go"].matching == "fuzzy"


def test_enabled_ontology_layers_helper():
    """`_enabled_ontology_layers()` returns the names of currently-on
    ontology flags. Used by the validators + downstream callers."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True, ontology_mesh=True, ontology_go=False),
        entities=EntityConfig(extractor="llm"),
    )
    assert config._enabled_ontology_layers() == ["mesh"]


def test_enabled_ontology_layers_empty_when_none_on():
    config = CorpusConfig()
    assert config._enabled_ontology_layers() == []


# ---- L8 triples layer ----


def test_load_corpus_config_triples_layer_default_off(tmp_path: Path):
    """`triples` defaults to False - existing corpora don't silently
    start running L8 on upgrade."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nchunks = true\n")
    config = load_corpus_config(toml_path)
    assert config.layers.triples is False


def test_load_corpus_config_triples_layer_on_requires_entities(tmp_path: Path):
    """`triples=true` without `entities=true` -> the LLM has no entity
    vocabulary to constrain to. Validator rejects at load time."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nchunks = true\ntriples = true\n")
    with pytest.raises(ValidationError, match="entities"):
        load_corpus_config(toml_path)


def test_load_corpus_config_triples_with_entities_loads_cleanly(tmp_path: Path):
    """The minimal valid L8-on config: triples=true + entities=true +
    [entities] block. No [triples] section needed at v1."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        '[layers]\nchunks = true\nentities = true\ntriples = true\n[entities]\nextractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.triples is True
    assert config.layers.entities is True


def test_triples_validator_runs_at_constructor_too():
    """The model validator catches direct CorpusConfig construction
    (not just toml load) - guards programmatic instantiation."""
    with pytest.raises(ValidationError, match="entities"):
        CorpusConfig(
            layers=LayerFlags(chunks=True, triples=True),
        )


# ---- L9 cross_doc layer ----


def test_load_corpus_config_cross_doc_layer_default_off(tmp_path: Path):
    """`cross_doc` defaults to False - existing corpora don't silently
    start writing :RELATED_TO edges on upgrade."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nchunks = true\n")
    config = load_corpus_config(toml_path)
    assert config.layers.cross_doc is False


def test_load_corpus_config_cross_doc_on_requires_entities(tmp_path: Path):
    """`cross_doc=true` without `entities=true` -> overlap query has
    nothing to compute. Validator rejects at load time."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text("[layers]\nchunks = true\ncross_doc = true\n")
    with pytest.raises(ValidationError, match="entities"):
        load_corpus_config(toml_path)


def test_load_corpus_config_cross_doc_with_entities_loads_cleanly(
    tmp_path: Path,
):
    """The minimal valid L9-on config."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "cross_doc = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.cross_doc is True
    assert config.layers.entities is True


def test_cross_doc_validator_runs_at_constructor_too():
    with pytest.raises(ValidationError, match="entities"):
        CorpusConfig(
            layers=LayerFlags(chunks=True, cross_doc=True),
        )


def test_cross_doc_and_triples_can_coexist(tmp_path: Path):
    """Both L8 and L9 layers can be enabled together - they're
    orthogonal (triples adds typed edges between entities; cross_doc
    adds undirected edges between docs based on shared entities)."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "triples = true\n"
        "cross_doc = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.triples is True
    assert config.layers.cross_doc is True


# ---- L7 xrefs + L10 cross_doc_xrefs layer flags ----


def test_xrefs_default_is_none():
    """The 3-state `xrefs` flag defaults to `"none"` so existing
    corpora don't silently start extracting xrefs when the project is
    upgraded."""
    flags = LayerFlags()
    assert flags.xrefs == "none"


def test_xrefs_accepts_all_three_states():
    """Every documented state parses without error."""
    for value in ("none", "collect_only", "use"):
        flags = LayerFlags(xrefs=value)
        assert flags.xrefs == value


def test_xrefs_rejects_unknown_value():
    """Pydantic Literal enforces the 3-state set; other strings error."""
    with pytest.raises(ValidationError):
        LayerFlags(xrefs="yes")
    with pytest.raises(ValidationError):
        LayerFlags(xrefs="")
    with pytest.raises(ValidationError):
        LayerFlags(xrefs="NONE")  # case-sensitive


def test_xrefs_round_trips_through_toml(tmp_path: Path):
    """Reading `xrefs = \"use\"` from corpus.toml lands on
    `config.layers.xrefs == \"use\"`."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        'xrefs = "collect_only"\n'
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.layers.xrefs == "collect_only"


def test_cross_doc_xrefs_default_is_false():
    flags = LayerFlags()
    assert flags.cross_doc_xrefs is False


def test_xrefs_without_any_ontology_is_silently_allowed():
    """Setting `xrefs="use"` without an ontology layer is harmless
    (no-op until ontologies ship). NOT an error - lets users prepare
    the flag before importing."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, xrefs="use"),
    )
    assert config.layers.xrefs == "use"


# ---- L9 + L10 threshold config classes ----


def test_cross_doc_config_default_threshold():
    """Default threshold is 2."""
    cfg = CrossDocConfig()
    assert cfg.threshold == 2


def test_cross_doc_xrefs_config_default_threshold():
    cfg = CrossDocXrefsConfig()
    assert cfg.threshold == 2


def test_cross_doc_config_rejects_threshold_zero():
    """`ge=1` enforced on both config models - threshold < 1 produces
    no edges (the predicate becomes trivially true), useless layer."""
    with pytest.raises(ValidationError):
        CrossDocConfig(threshold=0)
    with pytest.raises(ValidationError):
        CrossDocConfig(threshold=-1)


def test_cross_doc_xrefs_config_rejects_threshold_zero():
    with pytest.raises(ValidationError):
        CrossDocXrefsConfig(threshold=0)
    with pytest.raises(ValidationError):
        CrossDocXrefsConfig(threshold=-1)


def test_cross_doc_config_round_trips_through_toml(tmp_path: Path):
    """`[cross_doc]` block with `threshold = 5` lands on
    `config.cross_doc.threshold == 5`."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "cross_doc = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
        "[cross_doc]\n"
        "threshold = 5\n"
    )
    config = load_corpus_config(toml_path)
    assert config.cross_doc is not None
    assert config.cross_doc.threshold == 5


def test_cross_doc_xrefs_config_round_trips_through_toml(tmp_path: Path):
    """`[cross_doc_xrefs]` block with `threshold = 3` lands on
    `config.cross_doc_xrefs.threshold == 3`. Layer dependency is
    satisfied with entities=true + xrefs="use" + at least one ontology."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        'xrefs = "use"\n'
        "cross_doc_xrefs = true\n"
        "ontology_mesh = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
        "[cross_doc_xrefs]\n"
        "threshold = 3\n"
    )
    config = load_corpus_config(toml_path)
    assert config.cross_doc_xrefs is not None
    assert config.cross_doc_xrefs.threshold == 3


# ---- Auto-populate threshold configs ----


def test_cross_doc_config_auto_populates_when_layer_on(tmp_path: Path):
    """Enabling `layers.cross_doc=true` without the `[cross_doc]`
    section auto-populates a default `CrossDocConfig()` - so consumers
    can always read `config.cross_doc.threshold` after validation."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        "cross_doc = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.cross_doc is not None
    assert config.cross_doc.threshold == 2


def test_cross_doc_xrefs_config_auto_populates_when_layer_on(tmp_path: Path):
    """Enabling `layers.cross_doc_xrefs=true` without the
    `[cross_doc_xrefs]` section auto-populates a default
    `CrossDocXrefsConfig()`."""
    toml_path = tmp_path / "corpus.toml"
    toml_path.write_text(
        "[layers]\n"
        "chunks = true\n"
        "entities = true\n"
        'xrefs = "use"\n'
        "cross_doc_xrefs = true\n"
        "ontology_mesh = true\n"
        "[entities]\n"
        'extractor = "llm"\n'
    )
    config = load_corpus_config(toml_path)
    assert config.cross_doc_xrefs is not None
    assert config.cross_doc_xrefs.threshold == 2


def test_cross_doc_config_stays_none_when_layer_off():
    """When the layer is off, no auto-populate happens - the config
    stays None so consumers can distinguish 'user opted out' from
    'user accepted defaults'."""
    config = CorpusConfig(layers=LayerFlags(chunks=True))
    assert config.cross_doc is None
    assert config.cross_doc_xrefs is None


# ---- L10 dependency validators ----


def test_cross_doc_xrefs_true_requires_entities():
    """L10 walks Doc->Chunk->Entity->CANONICAL_TO->OntologyTerm; without
    entities the path collapses. Validator catches at config load."""
    with pytest.raises(ValidationError) as exc:
        CorpusConfig(
            layers=LayerFlags(
                chunks=True,
                cross_doc_xrefs=True,
                xrefs="use",
            ),
        )
    assert "layers.entities=true" in str(exc.value)


def test_cross_doc_xrefs_true_requires_xrefs_use():
    """xrefs=\"none\": no :<X>_XREF edges exist, equivalence collapses
    to identity-only. Validator errors."""
    with pytest.raises(ValidationError) as exc:
        CorpusConfig(
            layers=LayerFlags(
                chunks=True,
                entities=True,
                cross_doc_xrefs=True,
                xrefs="none",
            ),
            entities=EntityConfig(extractor="llm"),
        )
    assert 'xrefs="use"' in str(exc.value)


def test_cross_doc_xrefs_true_rejects_xrefs_collect_only():
    """xrefs=\"collect_only\": dangling_xrefs property exists but
    edges don't yet - L10 still has no edges to traverse. Validator
    errors with the same message as the 'none' case."""
    with pytest.raises(ValidationError) as exc:
        CorpusConfig(
            layers=LayerFlags(
                chunks=True,
                entities=True,
                cross_doc_xrefs=True,
                xrefs="collect_only",
            ),
            entities=EntityConfig(extractor="llm"),
        )
    assert 'xrefs="use"' in str(exc.value)


def test_cross_doc_xrefs_true_with_full_dependency_chain_accepts():
    """All preconditions met: entities=true + xrefs=use + at least one
    ontology + EntityConfig present. Validates cleanly."""
    config = CorpusConfig(
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            xrefs="use",
            cross_doc_xrefs=True,
        ),
        entities=EntityConfig(extractor="llm"),
    )
    assert config.layers.cross_doc_xrefs is True
    assert config.layers.xrefs == "use"
    # Auto-populated threshold available.
    assert config.cross_doc_xrefs is not None
    assert config.cross_doc_xrefs.threshold == 2


def test_cross_doc_xrefs_false_skips_dependency_check():
    """Layer flag off: xrefs can be anything, entities can be off.
    The validator only fires when the layer is on."""
    # All-off corpus.
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, cross_doc_xrefs=False),
    )
    assert config.layers.cross_doc_xrefs is False


# ---- figures directory (multimodal path derivation) ----
#
# The B2 rework moved figures from INSIDE lancedb to BESIDE it, at
# `<corpus>/figures/<doc_id>/`. These pin that location so a regression
# back to the old `<lancedb>/figures/` layout is caught.


def test_corpus_folder_is_toml_parent(tmp_path):
    toml = tmp_path / "mycorpus" / "corpus.toml"
    assert corpus_folder(toml) == tmp_path / "mycorpus"


def test_corpus_figures_dir_is_sibling_of_lancedb(tmp_path):
    """Figures live at `<corpus>/figures/<doc_id>/` — a sibling of
    `<corpus>/lancedb/`, NOT nested inside the LanceDB directory."""
    toml = tmp_path / "mycorpus" / "corpus.toml"
    d = corpus_figures_dir(toml, "doc-123")
    assert d == tmp_path / "mycorpus" / "figures" / "doc-123"
    # Explicitly NOT under a lancedb/ segment (the pre-2026-07-04 layout).
    assert "lancedb" not in d.parts


def test_corpus_figures_dir_created_idempotently(tmp_path):
    """The dir is mkdir'd on call (parents + exist_ok) so callers can
    write PNGs immediately; a second call on the same doc_id is a no-op."""
    toml = tmp_path / "c" / "corpus.toml"
    d1 = corpus_figures_dir(toml, "d1")
    assert d1.is_dir()
    d2 = corpus_figures_dir(toml, "d1")  # must not raise
    assert d1 == d2
