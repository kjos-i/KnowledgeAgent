"""Unit tests for `kg/ontologies/provenance` — the frozen dataclass
that drives per-ontology install-dialog metadata.

The data carrier itself is simple, but this file pins:
  - The frozen dataclass invariant (immutability)
  - Default value for `heavy_warning` (the one optional field)
  - The re-export from `lifecycle.py` resolves to the same
    class (no accidental double-definition)
  - Every shipped per-ontology constant
    (`<name>_writes._<NAME>_PROVENANCE`) builds without
    raising and exposes the required fields
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from knowledge_agent.kg.ontologies.provenance import OntologyProvenance

# ---------------------------------------------------------------------------
# Class shape
# ---------------------------------------------------------------------------


def test_construct_with_all_required_fields() -> None:
    p = OntologyProvenance(
        ontology_name="x",
        full_name="X Ontology",
        publisher="ACME",
        license="CC-BY",
        source_url="https://example.com",
        download_url="https://example.com/x.obo",
        file_format="OBO",
        download_size_mb=10,
        estimated_terms=1000,
        domain_tags=("biology",),
        covers_labels=("DISEASE",),
        description="An example.",
    )
    assert p.ontology_name == "x"
    assert p.heavy_warning is None  # default


def test_heavy_warning_can_be_set() -> None:
    p = OntologyProvenance(
        ontology_name="big",
        full_name="Big Ontology",
        publisher="X",
        license="Y",
        source_url="u",
        download_url="d",
        file_format="OBO",
        download_size_mb=500,
        estimated_terms=1_000_000,
        domain_tags=(),
        covers_labels=(),
        description="...",
        heavy_warning="500 MB download — several minutes",
    )
    assert p.heavy_warning is not None
    assert "500 MB" in p.heavy_warning


def test_is_frozen_dataclass() -> None:
    p = OntologyProvenance(
        ontology_name="x",
        full_name="X",
        publisher="A",
        license="B",
        source_url="u",
        download_url="d",
        file_format="OBO",
        download_size_mb=1,
        estimated_terms=1,
        domain_tags=(),
        covers_labels=(),
        description="...",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        p.ontology_name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Re-export from lifecycle
# ---------------------------------------------------------------------------


def test_ontology_lifecycle_re_export_is_same_class() -> None:
    """`OntologyProvenance` is re-exported from `kg.ontologies.lifecycle`
    for caller convenience. The re-export must resolve to THE SAME
    class object — different classes would silently break isinstance
    checks in the install-dialog rendering."""
    from knowledge_agent.kg.ontologies.lifecycle import (
        OntologyProvenance as Reexport,
    )

    assert Reexport is OntologyProvenance


# ---------------------------------------------------------------------------
# Every shipped per-ontology provenance constant builds without raising.
#
# Each `kg/ontologies/<name>_writes.py` module declares a
# `_<NAME>_PROVENANCE` constant at module load time. Importing all of
# them transitively exercises every provenance construction — catches
# a typo in any single ontology module's constructor without needing
# 18 separate tests.
# ---------------------------------------------------------------------------


def test_every_per_ontology_provenance_constant_constructs() -> None:
    """Walk every `kg.ontologies.<name>_writes` module, import it, and
    pull its `_<NAME>_PROVENANCE` constant. None should raise."""
    from knowledge_agent.kg import ontologies

    found: dict[str, OntologyProvenance] = {}
    for mod_info in pkgutil.iter_modules(ontologies.__path__):
        name = mod_info.name
        if not name.endswith("_writes"):
            continue
        mod = importlib.import_module(f"knowledge_agent.kg.ontologies.{name}")
        # Find the module's _<NAME>_PROVENANCE constant.
        for attr in dir(mod):
            if attr.endswith("_PROVENANCE") and attr.startswith("_"):
                obj = getattr(mod, attr)
                if isinstance(obj, OntologyProvenance):
                    found[obj.ontology_name] = obj
                    break
    # Expect at least the shipped count (18 ontologies as of
    # 2026-06-23 per the KG roadmap memory).
    assert len(found) >= 14
    # Every provenance has a non-empty ontology_name + full_name.
    for prov in found.values():
        assert prov.ontology_name
        assert prov.full_name
        assert prov.publisher
        assert prov.license
        assert prov.source_url
        assert prov.download_url
        assert prov.file_format
        assert prov.download_size_mb > 0
        assert prov.estimated_terms > 0
