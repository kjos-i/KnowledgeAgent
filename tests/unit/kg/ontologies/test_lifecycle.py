"""Tests for kg.ontologies.lifecycle - plan/execute admin ops for ontology
import + delete.

End-to-end behaviour (real Neo4j + real NLM/OBO downloads) is exercised
by `scripts/smoke_pipeline.py` indirectly via the auto-on-first-ingest
path. These unit tests pin the plan/execute split + the dispatch via
`ONTOLOGY_REGISTRY` with everything mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.kg.ontologies.lifecycle import (
    DeleteOntologyPlan,
    DeleteOntologyResult,
    ImportOntologyPlan,
    ImportOntologyResult,
    InstallCrossDocXrefsPlan,
    InstallXrefsPlan,
    LinkOntologyPlan,
    LinkOntologyResult,
    OntologyProvenance,
    delete_ontology_execute,
    delete_ontology_plan,
    get_canonicalization_candidates,
    get_extractor_candidates,
    import_ontology_execute,
    import_ontology_plan,
    install_cross_doc_xrefs_plan,
    install_xrefs_plan,
    link_ontology_execute,
    link_ontology_plan,
)

# ---- ImportOntologyPlan summary ----


def test_import_plan_summary_when_already_imported():
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=True,
        download_size_mb=115,
    )
    assert "already imported" in plan.summary.lower()


def test_import_plan_summary_when_not_imported_mentions_size():
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=False,
        download_size_mb=115,
    )
    assert "115 MB" in plan.summary
    assert "Import mesh" in plan.summary


# ---- import_ontology_plan ----


async def test_import_ontology_plan_raises_on_unknown_ontology():
    with pytest.raises(ValueError):
        await import_ontology_plan("not-a-real-ontology")


async def test_import_ontology_plan_marks_already_imported_when_kg_has_it():
    """is_imported_fn returns True -> plan flags already_imported."""
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "term_label": "MeSHTerm",
            "is_imported_fn": AsyncMock(return_value=True),
            "import_fn": AsyncMock(return_value=True),
            "delete_fn": AsyncMock(return_value=True),
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await import_ontology_plan("mesh")

    assert plan.already_imported is True
    assert plan.download_size_mb == 115


async def test_import_ontology_plan_marks_not_imported_when_kg_has_none():
    kg_mock = MagicMock()
    fake_registry = {
        "go": {
            "term_label": "GOTerm",
            "is_imported_fn": AsyncMock(return_value=False),
            "import_fn": AsyncMock(return_value=True),
            "delete_fn": AsyncMock(return_value=True),
            "download_size_mb": 200,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await import_ontology_plan("go")

    assert plan.already_imported is False


# ---- import_ontology_execute ----


async def test_import_ontology_execute_no_op_when_already_imported():
    """already_imported=True -> did_import False, import_ok True (idempotent)."""
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=True,
        download_size_mb=115,
    )
    import_fn = AsyncMock(return_value=True)
    fake_registry = {"mesh": {"import_fn": import_fn, "download_size_mb": 115}}
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client"),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await import_ontology_execute(plan)

    import_fn.assert_not_called()
    assert result.did_import is False
    assert result.import_ok is True


async def test_import_ontology_execute_runs_import_fn_when_not_imported():
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=False,
        download_size_mb=115,
    )
    import_fn = AsyncMock(return_value=True)
    fake_registry = {"mesh": {"import_fn": import_fn, "download_size_mb": 115}}
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=MagicMock()),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await import_ontology_execute(plan)

    import_fn.assert_called_once()
    assert result.did_import is True
    assert result.import_ok is True


async def test_import_ontology_execute_propagates_failure_from_import_fn():
    """Under typed-errors, the import_fn raises on failure; the
    execute boundary catches and reports `import_ok=False` plus a
    typed `import_error` carrying the message + exception class."""
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=False,
        download_size_mb=115,
    )

    def _failing_import(client, **kw):
        raise RuntimeError("network down")

    fake_registry = {
        "mesh": {
            "import_fn": _failing_import,
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=MagicMock()),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await import_ontology_execute(plan)

    assert result.did_import is True
    assert result.import_ok is False
    assert result.import_error is not None
    assert "network down" in result.import_error.message
    assert result.import_error.exception_type == "builtins.RuntimeError"


async def test_import_ontology_execute_returns_result_dataclass():
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=True,
        download_size_mb=115,
    )
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client"),
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY",
            {"mesh": {"import_fn": AsyncMock(return_value=True), "download_size_mb": 115}},
        ),
    ):
        result = await import_ontology_execute(plan)
    assert isinstance(result, ImportOntologyResult)


# ---- LinkOntologyPlan summary ----


def test_link_plan_summary_when_not_imported_redirects_to_install():
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        scope="global",
        matching_strategy="exact",
    )
    s = plan.summary.lower()
    assert "not imported" in s
    assert "install" in s


def test_link_plan_summary_global_scope_mentions_globally():
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        scope="global",
        matching_strategy="exact",
    )
    s = plan.summary
    assert "globally" in s
    assert "mesh" in s.lower()
    assert "exact" in s


def test_link_plan_summary_doc_scope_mentions_doc_id():
    plan = LinkOntologyPlan(
        ontology_name="go",
        is_imported=True,
        scope="abc123",
        matching_strategy="fuzzy",
    )
    s = plan.summary
    assert "abc123" in s
    assert "fuzzy" in s


# ---- link_ontology_plan ----


async def test_link_ontology_plan_raises_on_unknown_ontology():
    with pytest.raises(ValueError):
        await link_ontology_plan("not-a-real-ontology")


async def test_link_ontology_plan_raises_on_bad_matching_strategy():
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=True),
        },
    }
    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY",
            fake_registry,
        ),
        pytest.raises(ValueError),
    ):
        await link_ontology_plan("mesh", matching_strategy="nope")


async def test_link_ontology_plan_marks_is_imported_true_when_terms_present():
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=True),
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await link_ontology_plan("mesh")

    assert plan.is_imported is True


async def test_link_ontology_plan_marks_is_imported_false_when_no_terms():
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=False),
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await link_ontology_plan("mesh")

    assert plan.is_imported is False


async def test_link_ontology_plan_defaults_to_global_scope_and_exact_matching():
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=True),
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await link_ontology_plan("mesh")

    assert plan.scope == "global"
    assert plan.matching_strategy == "exact"


async def test_link_ontology_plan_accepts_doc_id_scope_and_fuzzy_matching():
    kg_mock = MagicMock()
    fake_registry = {
        "go": {
            "is_imported_fn": AsyncMock(return_value=True),
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await link_ontology_plan(
            "go",
            matching_strategy="fuzzy",
            scope="doc-xyz",
        )

    assert plan.scope == "doc-xyz"
    assert plan.matching_strategy == "fuzzy"


# ---- link_ontology_execute ----


async def test_link_ontology_execute_no_op_when_not_imported():
    """is_imported=False -> link_ok=False, no link_entities call."""
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        scope="global",
        matching_strategy="exact",
    )
    kg_mock = MagicMock()
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=kg_mock,
    ):
        result = await link_ontology_execute(plan)

    kg_mock.link_entities_to_ontology.assert_not_called()
    assert result.n_links_written == 0
    assert result.link_ok is False


async def test_link_ontology_execute_passes_none_doc_id_for_global_scope():
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        scope="global",
        matching_strategy="exact",
    )
    kg_mock = MagicMock()
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=42)
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=kg_mock,
    ):
        result = await link_ontology_execute(plan)

    kg_mock.link_entities_to_ontology.assert_called_once_with(
        "mesh",
        "exact",
        doc_id=None,
    )
    assert result.n_links_written == 42
    assert result.link_ok is True


async def test_link_ontology_execute_passes_doc_id_when_scope_is_doc_id():
    plan = LinkOntologyPlan(
        ontology_name="go",
        is_imported=True,
        scope="abc123",
        matching_strategy="fuzzy",
    )
    kg_mock = MagicMock()
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=7)
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=kg_mock,
    ):
        result = await link_ontology_execute(plan)

    kg_mock.link_entities_to_ontology.assert_called_once_with(
        "go",
        "fuzzy",
        doc_id="abc123",
    )
    assert result.n_links_written == 7
    assert result.scope == "abc123"


async def test_link_ontology_execute_zero_links_is_still_link_ok():
    """Empty match set is a valid outcome, not a failure."""
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        scope="global",
        matching_strategy="exact",
    )
    kg_mock = MagicMock()
    kg_mock.link_entities_to_ontology = AsyncMock(return_value=0)
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=kg_mock,
    ):
        result = await link_ontology_execute(plan)

    assert result.n_links_written == 0
    assert result.link_ok is True


async def test_link_ontology_execute_returns_result_dataclass():
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        scope="global",
        matching_strategy="exact",
    )
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=MagicMock(),
    ):
        result = await link_ontology_execute(plan)
    assert isinstance(result, LinkOntologyResult)


async def test_link_ontology_execute_catches_linking_pass_failure():
    """Under typed-errors, the linking pass raises on Cypher failure;
    the execute boundary catches and reports `link_ok=False` plus a
    typed `link_error` so the UI dialog can surface the cause."""
    plan = LinkOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        scope="global",
        matching_strategy="exact",
    )
    kg_mock = MagicMock()
    kg_mock.link_entities_to_ontology = AsyncMock(side_effect=RuntimeError("cypher boom"))
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
        return_value=kg_mock,
    ):
        result = await link_ontology_execute(plan)
    assert result.link_ok is False
    assert result.n_links_written == 0
    assert result.link_error is not None
    assert "cypher boom" in result.link_error.message
    assert result.link_error.exception_type == "builtins.RuntimeError"


# ---- DeleteOntologyPlan summary ----


def test_delete_plan_summary_when_not_imported_is_noop_message():
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        n_terms=0,
        n_canonical_links=0,
    )
    assert "not imported" in plan.summary.lower()


def test_delete_plan_summary_mentions_terms_and_links():
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        n_terms=30142,
        n_canonical_links=18,
    )
    s = plan.summary
    assert "30142" in s
    assert "18" in s
    assert "mesh" in s.lower()


# ---- delete_ontology_plan ----


async def test_delete_ontology_plan_raises_on_unknown_ontology():
    with pytest.raises(ValueError):
        await delete_ontology_plan("not-a-real-ontology")


async def test_delete_ontology_plan_skips_counts_when_not_imported():
    """is_imported_fn=False -> don't even run the count queries."""
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=False),
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await delete_ontology_plan("mesh")

    assert plan.is_imported is False
    assert plan.n_terms == 0
    assert plan.n_canonical_links == 0
    kg_mock.count_ontology_terms.assert_not_called()
    kg_mock.count_canonical_links.assert_not_called()


async def test_delete_ontology_plan_populates_counts_when_imported():
    kg_mock = MagicMock()
    kg_mock.count_ontology_terms = AsyncMock(return_value=30142)
    kg_mock.count_canonical_links = AsyncMock(return_value=18)
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=True),
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await delete_ontology_plan("mesh")

    assert plan.is_imported is True
    assert plan.n_terms == 30142
    assert plan.n_canonical_links == 18


async def test_delete_ontology_plan_propagates_count_failure():
    """A count primitive raising -> propagates under typed-errors
    contract. The caller (Layer 3 GUI handler) decides how to surface."""
    kg_mock = MagicMock()
    kg_mock.count_ontology_terms = AsyncMock(side_effect=RuntimeError("cypher boom"))
    fake_registry = {
        "mesh": {
            "is_imported_fn": AsyncMock(return_value=True),
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
        pytest.raises(RuntimeError, match="cypher boom"),
    ):
        await delete_ontology_plan("mesh")


# ---- delete_ontology_execute ----


async def test_delete_ontology_execute_no_op_when_not_imported():
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        n_terms=0,
        n_canonical_links=0,
    )
    delete_fn = AsyncMock(return_value=True)
    fake_registry = {"mesh": {"delete_fn": delete_fn}}
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client"),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await delete_ontology_execute(plan)

    delete_fn.assert_not_called()
    assert result.did_delete is False
    assert result.delete_ok is True


async def test_delete_ontology_execute_runs_delete_fn_when_imported():
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        n_terms=100,
        n_canonical_links=5,
    )
    delete_fn = AsyncMock(return_value=True)
    fake_registry = {"mesh": {"delete_fn": delete_fn}}
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=MagicMock()),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await delete_ontology_execute(plan)

    delete_fn.assert_called_once()
    assert result.did_delete is True
    assert result.delete_ok is True


async def test_delete_ontology_execute_propagates_failure():
    """Under typed-errors, delete_fn raises on failure; the execute
    boundary catches and reports `delete_ok=False` plus a typed
    `delete_error` carrying the message + exception class."""
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=True,
        n_terms=100,
        n_canonical_links=5,
    )

    def _failing_delete(client):
        raise RuntimeError("cypher boom")

    fake_registry = {"mesh": {"delete_fn": _failing_delete}}
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=MagicMock()),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await delete_ontology_execute(plan)

    assert result.did_delete is True
    assert result.delete_ok is False
    assert result.delete_error is not None
    assert "cypher boom" in result.delete_error.message
    assert result.delete_error.exception_type == "builtins.RuntimeError"


async def test_delete_ontology_execute_returns_result_dataclass():
    plan = DeleteOntologyPlan(
        ontology_name="mesh",
        is_imported=False,
        n_terms=0,
        n_canonical_links=0,
    )
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client"),
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY",
            {
                "mesh": {
                    "delete_fn": AsyncMock(return_value=True),
                }
            },
        ),
    ):
        result = await delete_ontology_execute(plan)
    assert isinstance(result, DeleteOntologyResult)


# ---- ImportOntologyPlan + xrefs_mode threading ----


def test_import_plan_summary_with_xrefs_use_describes_resolved_edges():
    plan = ImportOntologyPlan(
        ontology_name="mondo",
        already_imported=False,
        download_size_mb=180,
        xrefs_mode="use",
    )
    summary = plan.summary
    assert "180 MB" in summary
    assert '"use"' in summary
    assert "dangling_xrefs" in summary
    assert ":<X>_XREF" in summary


def test_import_plan_summary_with_xrefs_collect_only_no_edge_writes():
    plan = ImportOntologyPlan(
        ontology_name="mondo",
        already_imported=False,
        download_size_mb=180,
        xrefs_mode="collect_only",
    )
    summary = plan.summary
    assert "collect_only" in summary
    assert "dangling_xrefs" in summary
    # NOT a "will write resolved edges" promise - backfill_xrefs is
    # mentioned as the materialisation route instead.
    assert "backfill_xrefs" in summary


def test_import_plan_summary_xrefs_none_matches_legacy():
    """`xrefs_mode="none"` (default) produces the legacy summary."""
    plan = ImportOntologyPlan(
        ontology_name="mondo",
        already_imported=False,
        download_size_mb=180,
    )
    summary = plan.summary
    assert "180 MB" in summary
    # No xref-specific clauses on the default summary.
    assert "dangling_xrefs" not in summary
    assert ":<X>_XREF" not in summary


async def test_import_ontology_plan_threads_xrefs_mode_into_dataclass():
    kg_mock = MagicMock()
    fake_registry = {
        "mesh": {
            "term_label": "MeSHTerm",
            "is_imported_fn": AsyncMock(return_value=False),
            "import_fn": AsyncMock(return_value=True),
            "delete_fn": AsyncMock(return_value=True),
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await import_ontology_plan("mesh", xrefs_mode="use")
    assert plan.xrefs_mode == "use"


async def test_import_ontology_execute_passes_xrefs_mode_to_import_fn():
    """The plan's `xrefs_mode` flows through to the per-ontology
    `import_fn` as the kwarg of the same name."""
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=False,
        download_size_mb=115,
        xrefs_mode="collect_only",
    )
    captured: dict = {}

    async def fake_import_fn(client, **kwargs):
        captured.update(kwargs)
        return True

    fake_registry = {
        "mesh": {
            "import_fn": fake_import_fn,
            "download_size_mb": 115,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=MagicMock()),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        result = await import_ontology_execute(plan)
    assert result.import_ok is True
    assert captured.get("xrefs_mode") == "collect_only"


# ---- InstallXrefsPlan ----


def test_install_xrefs_plan_validates_mode():
    """The dataclass `__post_init__` rejects unrecognised modes."""
    with pytest.raises(ValueError):
        InstallXrefsPlan(
            new_xrefs_mode="yes",
            n_imported_ontologies=0,
            n_dangling_xref_sources=0,
            n_existing_xref_edges=0,
        )


def test_install_xrefs_plan_summary_use_describes_immediate_writes():
    plan = InstallXrefsPlan(
        new_xrefs_mode="use",
        n_imported_ontologies=5,
        n_dangling_xref_sources=42,
        n_existing_xref_edges=100,
    )
    s = plan.summary
    assert '"use"' in s
    assert "5 of" in s  # n imported / 18
    assert "42" in s  # dangling count
    assert "100" in s  # existing edges
    assert "backfill_xrefs" in s
    assert "force-re-import" in s


def test_install_xrefs_plan_summary_collect_only_no_edge_writes():
    plan = InstallXrefsPlan(
        new_xrefs_mode="collect_only",
        n_imported_ontologies=2,
        n_dangling_xref_sources=0,
        n_existing_xref_edges=0,
    )
    s = plan.summary
    assert "collect_only" in s
    assert "will NOT" in s
    # collect_only's summary tells the user how to materialise later.
    assert "backfill_xrefs" in s


def test_install_xrefs_plan_summary_none_describes_disable():
    plan = InstallXrefsPlan(
        new_xrefs_mode="none",
        n_imported_ontologies=3,
        n_dangling_xref_sources=10,
        n_existing_xref_edges=50,
    )
    s = plan.summary
    assert '"none"' in s
    assert "skip" in s
    # Existing data NOT removed by this flip — the summary says so.
    assert "NOT removed" in s


async def test_install_xrefs_plan_factory_aggregates_counts_across_18():
    """`install_xrefs_plan` walks all 18 sub-labels via the diagnostics
    in `kg.ontologies.xrefs` and aggregates counts."""

    # Stub the count primitives to return predictable per-label counts.
    def fake_dangling(client, term_label):
        return {"MeSHTerm": 4, "GOTerm": 3}.get(term_label, 0)

    def fake_edges(client, term_label):
        # global counter (term_label=None)
        assert term_label is None
        return 17

    async def fake_is_imported_fn(client):
        return True  # treat every registry entry as imported

    fake_registry_entry = {
        "term_label": "MeSHTerm",
        "is_imported_fn": fake_is_imported_fn,
        "import_fn": AsyncMock(return_value=True),
        "delete_fn": AsyncMock(return_value=True),
        "download_size_mb": 115,
    }
    # Registry must map for every term_label so the factory's
    # is_imported probe always finds an entry.
    from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS

    fake_registry = {
        f"key_{lbl}": {**fake_registry_entry, "term_label": lbl} for lbl in ONTOLOGY_SUB_LABELS
    }
    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.count_dangling_xrefs",
            side_effect=fake_dangling,
        ),
        patch("knowledge_agent.kg.ontologies.lifecycle.count_xref_edges", side_effect=fake_edges),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
        # Rebuild the cached label->registry-key map after the patch.
        patch(
            "knowledge_agent.kg.ontologies.lifecycle._TERM_LABEL_TO_REGISTRY_KEY",
            {lbl: f"key_{lbl}" for lbl in ONTOLOGY_SUB_LABELS},
        ),
    ):
        plan = await install_xrefs_plan(MagicMock(), "use")
    assert plan.new_xrefs_mode == "use"
    assert plan.n_imported_ontologies == 18
    assert plan.n_dangling_xref_sources == 4 + 3  # MeSH + GO contribute
    assert plan.n_existing_xref_edges == 17


async def test_install_xrefs_plan_factory_rejects_unknown_mode():
    with pytest.raises(ValueError):
        await install_xrefs_plan(MagicMock(), "definitely_not_a_mode")


async def test_install_xrefs_plan_factory_fail_soft_on_dangling_count_error():
    """count_dangling_xrefs RAISES on a Cypher/driver error (it is typed
    `-> int` and never returns None). The factory must skip that sub-label and
    still build the plan from the other 17. Regression: the dead
    `if dangling is None` guard let the raised error crash the whole plan."""

    def fake_dangling(client, term_label):
        if term_label == "MeSHTerm":
            raise RuntimeError("cypher boom for MeSHTerm")
        return 0

    def fake_edges(client, term_label):
        return 0

    async def fake_is_imported_fn(client):
        return True

    fake_registry_entry = {
        "term_label": "MeSHTerm",
        "is_imported_fn": fake_is_imported_fn,
        "import_fn": AsyncMock(return_value=True),
        "delete_fn": AsyncMock(return_value=True),
        "download_size_mb": 1,
    }
    from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS

    fake_registry = {
        f"key_{lbl}": {**fake_registry_entry, "term_label": lbl} for lbl in ONTOLOGY_SUB_LABELS
    }
    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.count_dangling_xrefs",
            side_effect=fake_dangling,
        ),
        patch("knowledge_agent.kg.ontologies.lifecycle.count_xref_edges", side_effect=fake_edges),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
        patch(
            "knowledge_agent.kg.ontologies.lifecycle._TERM_LABEL_TO_REGISTRY_KEY",
            {lbl: f"key_{lbl}" for lbl in ONTOLOGY_SUB_LABELS},
        ),
    ):
        plan = await install_xrefs_plan(MagicMock(), "collect_only")
    # The plan still built (did NOT crash); MeSHTerm was skipped, so only the
    # other 17 imported sub-labels are counted.
    assert isinstance(plan, InstallXrefsPlan)
    assert plan.n_imported_ontologies == len(ONTOLOGY_SUB_LABELS) - 1


# ---- InstallCrossDocXrefsPlan ----


def test_install_cross_doc_xrefs_plan_summary_enabling_with_deps_met():
    plan = InstallCrossDocXrefsPlan(
        enabling=True,
        n_docs=120,
        n_existing_l10_edges=0,
        entities_layer_on=True,
        xrefs_layer_use=True,
    )
    s = plan.summary
    assert "Enable" in s
    assert "120" in s
    assert "RELATED_BY_XREF" in s
    assert "recompute_cross_doc_xrefs" in s


def test_install_cross_doc_xrefs_plan_summary_enabling_missing_deps():
    """When dependencies are NOT satisfied at flip time, the summary
    surfaces which ones are missing."""
    plan = InstallCrossDocXrefsPlan(
        enabling=True,
        n_docs=0,
        n_existing_l10_edges=0,
        entities_layer_on=False,
        xrefs_layer_use=False,
    )
    s = plan.summary
    assert "entities=true" in s
    assert 'xrefs="use"' in s


def test_install_cross_doc_xrefs_plan_summary_disabling():
    plan = InstallCrossDocXrefsPlan(
        enabling=False,
        n_docs=120,
        n_existing_l10_edges=37,
        entities_layer_on=True,
        xrefs_layer_use=True,
    )
    s = plan.summary
    assert "Disable" in s
    assert "37" in s
    # Disabling does NOT remove existing edges — the summary says so.
    assert "does NOT remove them" in s


async def test_install_cross_doc_xrefs_plan_factory_reads_live_counts():
    """The factory queries `:Document|:Artifact` count and the L10
    edge count via await session.run(); both come back as the plan's
    `n_docs` / `n_existing_l10_edges`."""
    sessions_made = []

    class FakeSession:
        def __init__(self):
            self.calls = []
            self._iter = iter(
                [
                    {"n": 73},  # count of focal nodes
                    {"n": 19},  # count of L10 edges
                ]
            )

        async def run(self, query, **params):
            self.calls.append(query)
            row = next(self._iter)

            class _R:
                def __init__(self, r):
                    self._r = r

                async def single(self):
                    return self._r

            return _R(row)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class FakeDriver:
        def __init__(self):
            self.sess = FakeSession()

        def session(self):
            sessions_made.append(self.sess)
            return self.sess

    class FakeClient:
        driver = FakeDriver()

    plan = await install_cross_doc_xrefs_plan(
        FakeClient(),
        enabling=True,
        entities_layer_on=True,
        xrefs_layer_use=True,
    )
    assert plan.n_docs == 73
    assert plan.n_existing_l10_edges == 19


async def test_install_cross_doc_xrefs_plan_factory_fail_soft_on_count_error():
    """If both count primitives raise, the plan still builds with
    n_docs=0 and n_existing_l10_edges=0 (graceful degradation)."""

    class FakeDriver:
        def session(self):
            raise RuntimeError("driver down")

    class FakeClient:
        driver = FakeDriver()

    plan = await install_cross_doc_xrefs_plan(
        FakeClient(),
        enabling=True,
        entities_layer_on=True,
        xrefs_layer_use=True,
    )
    assert plan.n_docs == 0
    assert plan.n_existing_l10_edges == 0


# ---- OntologyProvenance dataclass + rich summary rendering ----


def _provenance_kwargs(**overrides):
    """Stub provenance kwargs covering every required field. Tests
    override individual fields by passing them in."""
    defaults = dict(
        ontology_name="stub",
        full_name="Stub Ontology",
        publisher="Stub Org",
        license="Stub License (free use)",
        source_url="https://example.com/stub",
        download_url="https://example.com/stub.obo",
        file_format="OBO",
        download_size_mb=50,
        estimated_terms=12345,
        domain_tags=("biology",),
        covers_labels=("CHEMICAL",),
        description="A stub ontology for testing.",
        heavy_warning=None,
    )
    defaults.update(overrides)
    return defaults


def test_ontology_provenance_is_frozen():
    """Plan dialogs must not be able to mutate provenance between
    dialog render and execute. Frozen dataclass enforces that."""
    p = OntologyProvenance(**_provenance_kwargs())
    with pytest.raises(Exception):
        # dataclasses.FrozenInstanceError - too specific to type-check.
        p.publisher = "different"  # type: ignore[misc]


def test_ontology_provenance_carries_every_documented_field():
    """Construction with the full field set succeeds and round-trips."""
    p = OntologyProvenance(
        ontology_name="mesh",
        full_name="Medical Subject Headings",
        publisher="NLM",
        license="NLM Terms",
        source_url="https://www.nlm.nih.gov/mesh/meshhome.html",
        download_url="https://example/mesh.nt.gz",
        file_format="N-Triples (gzipped)",
        download_size_mb=115,
        estimated_terms=30000,
        domain_tags=("medicine", "biology"),
        covers_labels=("DISEASE", "CHEMICAL"),
        description="NLM's biomedical thesaurus.",
        heavy_warning="memory-heavy",
    )
    assert p.ontology_name == "mesh"
    assert p.full_name == "Medical Subject Headings"
    assert p.publisher == "NLM"
    assert p.license == "NLM Terms"
    assert p.source_url == "https://www.nlm.nih.gov/mesh/meshhome.html"
    assert p.download_url == "https://example/mesh.nt.gz"
    assert p.file_format == "N-Triples (gzipped)"
    assert p.download_size_mb == 115
    assert p.estimated_terms == 30000
    assert p.domain_tags == ("medicine", "biology")
    assert p.covers_labels == ("DISEASE", "CHEMICAL")
    assert p.description == "NLM's biomedical thesaurus."
    assert p.heavy_warning == "memory-heavy"


def test_ontology_provenance_heavy_warning_defaults_none():
    """Most ontologies have no heavy warning; default is None."""
    p = OntologyProvenance(**_provenance_kwargs())
    assert p.heavy_warning is None


# ---- ImportOntologyPlan rich summary when provenance is present ----


def test_import_plan_legacy_summary_when_provenance_is_none():
    """Provenance=None (default) preserves the pre-provenance
    one-liner — back-compat for tests + transitional code."""
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=False,
        download_size_mb=115,
    )
    assert plan.summary == "Import mesh (~115 MB download)."


def test_import_plan_rich_summary_when_provenance_present():
    """Provenance set -> summary picks up publisher / license /
    source / domain / covers_labels / description, replacing the
    legacy bare line."""
    p = OntologyProvenance(
        **_provenance_kwargs(
            ontology_name="mondo",
            full_name="Mondo Disease Ontology",
            publisher="Mondo Initiative + Monarch Initiative",
            license="CC0 1.0",
            source_url="https://mondo.monarchinitiative.org/",
            download_url="https://example/mondo.obo",
            file_format="OBO",
            download_size_mb=30,
            estimated_terms=24000,
            domain_tags=("medicine",),
            covers_labels=("DISEASE",),
            description="Integrated disease ontology across DOID, OMIM, Orphanet, etc.",
        )
    )
    plan = ImportOntologyPlan(
        ontology_name="mondo",
        already_imported=False,
        download_size_mb=30,
        provenance=p,
    )
    s = plan.summary
    assert "Mondo Disease Ontology" in s
    assert "Mondo Initiative + Monarch Initiative" in s
    assert "CC0 1.0" in s
    assert "https://mondo.monarchinitiative.org/" in s
    assert "OBO" in s
    assert "24,000" in s  # formatted with commas
    assert "medicine" in s
    assert "DISEASE" in s
    assert "Integrated disease ontology" in s


def test_import_plan_rich_summary_includes_heavy_warning_when_present():
    p = OntologyProvenance(
        **_provenance_kwargs(
            ontology_name="ncbitaxon",
            full_name="NCBI Taxonomy",
            heavy_warning="~2.74M classes / ~440 MB / several GB RAM at import",
        )
    )
    plan = ImportOntologyPlan(
        ontology_name="ncbitaxon",
        already_imported=False,
        download_size_mb=440,
        provenance=p,
    )
    s = plan.summary
    assert "WARNING" in s
    assert "2.74M" in s


def test_import_plan_already_imported_short_circuits_even_with_provenance():
    """The 'already imported' check beats the rich summary — no point
    rendering a long block for a no-op."""
    p = OntologyProvenance(**_provenance_kwargs(ontology_name="mesh"))
    plan = ImportOntologyPlan(
        ontology_name="mesh",
        already_imported=True,
        download_size_mb=115,
        provenance=p,
    )
    assert "already imported" in plan.summary.lower()
    # Provenance fields don't bleed into the short-circuited summary.
    assert "Publisher" not in plan.summary
    assert "License" not in plan.summary


def test_import_plan_rich_summary_composes_with_xrefs_use_clause():
    """xrefs_mode='use' appends its clause AFTER the rich block."""
    p = OntologyProvenance(**_provenance_kwargs(ontology_name="chebi"))
    plan = ImportOntologyPlan(
        ontology_name="chebi",
        already_imported=False,
        download_size_mb=50,
        xrefs_mode="use",
        provenance=p,
    )
    s = plan.summary
    # Both the rich provenance content AND the xrefs clause appear.
    assert "Publisher" in s
    assert "License" in s
    assert '"use"' in s
    assert ":<X>_XREF" in s


# ---- import_ontology_plan factory reads provenance from registry ----


async def test_import_ontology_plan_factory_reads_provenance_from_registry():
    """When the registry entry carries `"provenance"`, the factory
    threads it onto the returned ImportOntologyPlan."""
    kg_mock = MagicMock()
    p = OntologyProvenance(**_provenance_kwargs(ontology_name="mesh"))
    fake_registry = {
        "mesh": {
            "term_label": "MeSHTerm",
            "is_imported_fn": AsyncMock(return_value=False),
            "import_fn": AsyncMock(return_value=True),
            "delete_fn": AsyncMock(return_value=True),
            "download_size_mb": 115,
            "provenance": p,
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await import_ontology_plan("mesh")
    assert plan.provenance is p


async def test_import_ontology_plan_factory_handles_registry_without_provenance():
    """Registry entries missing `"provenance"` (transitional during the
    18-ontology rollout) produce a plan with provenance=None — summary
    falls back to legacy one-liner."""
    kg_mock = MagicMock()
    fake_registry = {
        "go": {
            "term_label": "GOTerm",
            "is_imported_fn": AsyncMock(return_value=False),
            "import_fn": AsyncMock(return_value=True),
            "delete_fn": AsyncMock(return_value=True),
            "download_size_mb": 200,
            # NO "provenance" key
        },
    }
    with (
        patch("knowledge_agent.kg.ontologies.lifecycle.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.kg.ontologies.lifecycle.ONTOLOGY_REGISTRY", fake_registry),
    ):
        plan = await import_ontology_plan("go")
    assert plan.provenance is None
    # Legacy summary still works.
    assert plan.summary == "Import go (~200 MB download)."


# ---- MeSH pilot — the one ontology wired in step 1 ----


def test_mesh_pilot_provenance_constant_exposed_via_module():
    """The MeSH module exports a `_MESH_PROVENANCE` constant with
    real values that match the registry."""
    from knowledge_agent.kg.ontologies import mesh_writes

    p = mesh_writes._MESH_PROVENANCE
    assert isinstance(p, OntologyProvenance)
    assert p.ontology_name == "mesh"
    assert p.full_name == "Medical Subject Headings"
    assert "NLM" in p.publisher
    assert p.file_format == "N-Triples (gzipped)"
    assert "medicine" in p.domain_tags
    assert "DISEASE" in p.covers_labels


def test_mesh_pilot_registry_entry_carries_provenance():
    """The live registry routes MeSH through the pilot provenance."""
    from knowledge_agent.kg.ontologies import mesh_writes
    from knowledge_agent.kg.ontologies.linking import (
        ONTOLOGY_REGISTRY,
    )

    entry = ONTOLOGY_REGISTRY["mesh"]
    assert entry.get("provenance") is mesh_writes._MESH_PROVENANCE


def test_every_shipped_ontology_has_registry_provenance():
    """After step 2, every registry entry must carry a `provenance`
    pointing at a real `OntologyProvenance`. The install dialog can
    rely on the field being present for any shipped ontology."""
    from knowledge_agent.kg.ontologies.linking import (
        ONTOLOGY_REGISTRY,
    )

    for name, entry in ONTOLOGY_REGISTRY.items():
        provenance = entry.get("provenance")
        assert isinstance(provenance, OntologyProvenance), (
            f"{name}: registry entry is missing provenance (expected OntologyProvenance instance)."
        )
        # The provenance's own name field matches the registry key.
        assert provenance.ontology_name == name, (
            f"{name}: provenance.ontology_name = "
            f"{provenance.ontology_name!r}; should equal registry key."
        )


def test_every_shipped_ontology_provenance_has_required_fields():
    """Soft assertions on field shape: required strings non-empty,
    sizes positive, domain_tags non-empty."""
    from knowledge_agent.kg.ontologies.linking import (
        ONTOLOGY_REGISTRY,
    )

    for name, entry in ONTOLOGY_REGISTRY.items():
        p = entry["provenance"]
        assert p.full_name, f"{name}: full_name empty"
        assert p.publisher, f"{name}: publisher empty"
        assert p.license, f"{name}: license empty"
        assert p.source_url.startswith("http"), (
            f"{name}: source_url not URL-shaped: {p.source_url!r}"
        )
        assert p.download_url.startswith("http"), (
            f"{name}: download_url not URL-shaped: {p.download_url!r}"
        )
        assert p.file_format, f"{name}: file_format empty"
        assert p.download_size_mb > 0, f"{name}: download_size_mb must be > 0"
        assert p.estimated_terms > 0, f"{name}: estimated_terms must be > 0"
        assert p.domain_tags, f"{name}: domain_tags must be non-empty"
        assert p.description, f"{name}: description empty"


def test_ncbitaxon_and_dron_have_heavy_warnings():
    """The two memory-heavy ontologies carry a heavy_warning so the
    install dialog can surface the RAM caveat prominently."""
    from knowledge_agent.kg.ontologies.linking import (
        ONTOLOGY_REGISTRY,
    )

    assert ONTOLOGY_REGISTRY["ncbitaxon"]["provenance"].heavy_warning
    assert ONTOLOGY_REGISTRY["dron"]["provenance"].heavy_warning


def test_go_domain_tags_dropped_pharmacology():
    """GO covers biology / chemistry / proteins; pharmacology was
    dropped in step 2 since drug pharmacology belongs to DRON/ChEBI."""
    from knowledge_agent.kg.ontologies import go_writes

    assert go_writes.DOMAIN_TAGS == (
        "biology",
        "chemistry",
        "proteins",
    )
    assert go_writes._GO_PROVENANCE.domain_tags == (
        "biology",
        "chemistry",
        "proteins",
    )


async def test_mesh_pilot_factory_produces_rich_summary():
    """End-to-end against the live registry: `await import_ontology_plan("mesh")`
    builds a plan whose summary surfaces real MeSH provenance values.

    The registry's is_imported_fn captures the module-level function at
    import time; patching the module attribute later doesn't reach the
    captured reference. We swap out the registry entry's
    is_imported_fn directly to force the not-imported path."""
    from knowledge_agent.kg.ontologies.linking import (
        ONTOLOGY_REGISTRY,
    )

    # Preserve original, swap in a stub, restore after.
    async def _not_imported(_c):
        return False

    original_entry = dict(ONTOLOGY_REGISTRY["mesh"])
    ONTOLOGY_REGISTRY["mesh"]["is_imported_fn"] = _not_imported
    try:
        with patch(
            "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
            return_value=MagicMock(),
        ):
            plan = await import_ontology_plan("mesh")
    finally:
        ONTOLOGY_REGISTRY["mesh"] = original_entry
    s = plan.summary
    assert "Medical Subject Headings" in s
    assert "NLM" in s
    assert "DISEASE" in s


# ---- Cross-link helpers: get_canonicalization_candidates ----


def test_get_canonicalization_candidates_unknown_extractor_raises():
    with pytest.raises(ValueError):
        get_canonicalization_candidates("not-a-real-extractor")


def test_get_canonicalization_candidates_llm_returns_empty_dict():
    """LLM is open-vocabulary — no static mapping makes sense; helper
    returns {} so the dialog can render the runtime disclaimer."""
    assert get_canonicalization_candidates("llm") == {}


def test_get_canonicalization_candidates_gliner_biomed_routes_disease_to_disease_ontologies():
    """GLiNER-BioMed's DISEASE label maps to every ontology with
    DISEASE in its covers_labels: MONDO + MeSH + HPO + EFO (per the
    step 2 declarations)."""
    candidates = get_canonicalization_candidates("gliner_biomed")
    disease_targets = candidates["DISEASE"]
    assert "mondo" in disease_targets
    assert "mesh" in disease_targets
    assert "hpo" in disease_targets
    assert "efo" in disease_targets


def test_get_canonicalization_candidates_gliner_biomed_routes_chemical_to_chebi_and_mesh():
    candidates = get_canonicalization_candidates("gliner_biomed")
    chem_targets = candidates["CHEMICAL"]
    assert "chebi" in chem_targets
    assert "mesh" in chem_targets
    assert "dron" in chem_targets


def test_get_canonicalization_candidates_per_label_keys_sorted_alphabetically():
    """Each label's ontology-name tuple is alphabetically sorted for
    deterministic dialog rendering."""
    candidates = get_canonicalization_candidates("gliner_biomed")
    disease_targets = candidates["DISEASE"]
    assert list(disease_targets) == sorted(disease_targets)


def test_get_canonicalization_candidates_includes_labels_with_no_ontology_match():
    """A label that no shipped ontology covers (e.g. CELL_TYPE if no
    ontology declared it — wait, CL does. Try GLiNER's general
    labels: PERSON / ORGANIZATION have no biomedical ontology
    pairing). Helper still includes the label in the dict, with an
    empty tuple value — the dialog can say 'no shipped ontology
    covers this'."""
    candidates = get_canonicalization_candidates("gliner")
    # PERSON / ORGANIZATION / LOCATION never appear in any
    # ontology's covers_labels.
    assert candidates["PERSON"] == ()
    assert candidates["ORGANIZATION"] == ()
    assert candidates["LOCATION"] == ()


def test_get_canonicalization_candidates_hunflair2_5_label_set():
    """HunFlair2 emits exactly 5 labels — each gets a candidate
    entry (possibly empty)."""
    candidates = get_canonicalization_candidates("hunflair2")
    assert set(candidates) == {
        "DISEASE",
        "CHEMICAL",
        "GENE",
        "SPECIES",
        "CELL_LINE",
    }


# ---- Cross-link helpers: get_extractor_candidates ----


def test_get_extractor_candidates_unknown_ontology_raises():
    with pytest.raises(ValueError):
        get_extractor_candidates("not-a-real-ontology")


def test_get_extractor_candidates_mesh_pairs_with_biomedical_extractors():
    """MeSH's covers_labels include DISEASE + CHEMICAL + ANATOMY +
    GENE. Both GLiNER-BioMed and HunFlair2 should appear with
    matching labels."""
    candidates = get_extractor_candidates("mesh")
    by_name = {ext: labels for ext, labels in candidates}
    assert "gliner_biomed" in by_name
    assert "hunflair2" in by_name
    # GLiNER-BioMed has wider coverage (DISEASE / CHEMICAL / GENE /
    # ANATOMY all in its emitted_labels).
    assert "DISEASE" in by_name["gliner_biomed"]
    assert "CHEMICAL" in by_name["gliner_biomed"]
    assert "ANATOMY" in by_name["gliner_biomed"]


def test_get_extractor_candidates_sorted_by_overlap_size_desc():
    """When MeSH covers DISEASE + CHEMICAL + ANATOMY + GENE,
    gliner_biomed (4 matches) appears before hunflair2 (3 matches:
    DISEASE + CHEMICAL + GENE — no ANATOMY)."""
    candidates = get_extractor_candidates("mesh")
    if len(candidates) >= 2:
        # First entry has at least as many matches as the second.
        assert len(candidates[0][1]) >= len(candidates[1][1])


def test_get_extractor_candidates_eco_returns_empty_tuple():
    """ECO's covers_labels is empty (per step 2 — useful as
    provenance metadata, not for canonicalisation). No extractor
    pairings."""
    assert get_extractor_candidates("eco") == ()


def test_get_extractor_candidates_foodon_returns_empty_tuple():
    assert get_extractor_candidates("foodon") == ()


def test_get_extractor_candidates_chebi_includes_dron_too():
    """ChEBI covers CHEMICAL — all CHEMICAL-emitting extractors
    pair with it: GLiNER-BioMed + HunFlair2."""
    candidates = get_extractor_candidates("chebi")
    names = [ext for ext, _ in candidates]
    assert "gliner_biomed" in names
    assert "hunflair2" in names


def test_get_extractor_candidates_skips_llm_open_vocab():
    """LLM (`emitted_labels=None`) matches everything lexically at
    runtime — but listing it under every ontology would be noise.
    Helper explicitly skips it."""
    candidates = get_extractor_candidates("mesh")
    names = [ext for ext, _ in candidates]
    assert "llm" not in names


# ---- ImportOntologyPlan summary with cross-link surface ----


async def test_import_plan_rich_summary_surfaces_extractor_candidates():
    """When the plan carries extractor_candidates, the summary
    appends a 'Pairs well with extractors:' block."""
    from knowledge_agent.kg.ontologies.linking import ONTOLOGY_REGISTRY

    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
            return_value=MagicMock(),
        ),
        patch.dict(
            ONTOLOGY_REGISTRY["mesh"],
            {
                "is_imported_fn": AsyncMock(return_value=False),
            },
        ),
    ):
        plan = await import_ontology_plan("mesh")
    s = plan.summary
    assert "Pairs well with extractors" in s
    # At least one biomedical extractor name appears in the block.
    assert "gliner_biomed" in s or "hunflair2" in s


async def test_import_plan_rich_summary_handles_empty_extractor_candidates():
    """ECO has covers_labels=() so extractor_candidates is empty —
    summary surfaces the explanatory 'no extractor-pairable surface'
    note instead of dangling the Pairs-well-with block."""
    from knowledge_agent.kg.ontologies.linking import ONTOLOGY_REGISTRY

    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
            return_value=MagicMock(),
        ),
        patch.dict(
            ONTOLOGY_REGISTRY["eco"],
            {
                "is_imported_fn": AsyncMock(return_value=False),
            },
        ),
    ):
        plan = await import_ontology_plan("eco")
    s = plan.summary
    assert "Pairs well with" not in s
    assert "no extractor-pairable" in s


def test_import_plan_summary_no_extractor_pairs_when_covers_set_but_no_match():
    """Hypothetical: covers_labels declared but no shipped extractor
    emits matching labels. Summary says so explicitly (don't dangle
    'Pairs well with' on an empty block)."""
    p = OntologyProvenance(
        **_provenance_kwargs(
            ontology_name="hypothetical",
            full_name="Hypothetical",
            covers_labels=("UNICORN_LABEL",),  # no extractor emits this
        )
    )
    plan = ImportOntologyPlan(
        ontology_name="hypothetical",
        already_imported=False,
        download_size_mb=10,
        provenance=p,
        extractor_candidates=(),
    )
    s = plan.summary
    assert "No shipped NER extractor emits" in s


# ---- InstallExtractorPlan summary with cross-link surface ----


def test_install_extractor_plan_surfaces_canonicalisation_targets():
    """GLiNER-BioMed's install summary surfaces the per-label
    canonicalisation routes."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        install_extractor_plan,
    )

    plan = install_extractor_plan("gliner_biomed")
    s = plan.summary
    assert "Canonicalisation targets" in s
    # DISEASE routes to multiple disease ontologies.
    assert "DISEASE" in s
    assert "mondo" in s or "mesh" in s


def test_install_extractor_plan_marks_unmatched_label_explicitly():
    """GLiNER general emits PERSON which no shipped ontology covers
    — the summary explicitly says so per-label."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        install_extractor_plan,
    )

    plan = install_extractor_plan("gliner")
    s = plan.summary
    assert "PERSON" in s
    assert "no shipped ontology covers this" in s


def test_install_extractor_plan_llm_says_open_vocab_lexically():
    """LLM plan surfaces the 'any ontology lexically' line instead
    of a per-label mapping."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        install_extractor_plan,
    )

    plan = install_extractor_plan("llm")
    s = plan.summary
    assert "any ontology lexically" in s.lower() or "open vocabulary" in s.lower()


# ---- Factory threading ----


async def test_import_ontology_plan_threads_extractor_candidates():
    """Factory pre-computes extractor candidates and stores them on
    the plan."""
    from knowledge_agent.kg.ontologies.linking import ONTOLOGY_REGISTRY

    with (
        patch(
            "knowledge_agent.kg.ontologies.lifecycle.get_kg_client",
            return_value=MagicMock(),
        ),
        patch.dict(
            ONTOLOGY_REGISTRY["mondo"],
            {
                "is_imported_fn": AsyncMock(return_value=False),
            },
        ),
    ):
        plan = await import_ontology_plan("mondo")
    # MONDO covers DISEASE; at least one biomedical extractor should
    # match.
    assert len(plan.extractor_candidates) >= 1


def test_install_extractor_plan_threads_canonicalization_candidates():
    """Factory pre-computes per-label ontology candidates and stores
    them on the plan."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        install_extractor_plan,
    )

    plan = install_extractor_plan("gliner_biomed")
    assert plan.canonicalization_candidates is not None
    by_label = dict(plan.canonicalization_candidates)
    assert "DISEASE" in by_label


def test_install_extractor_plan_llm_canonicalization_candidates_is_none():
    """LLM keeps the None sentinel for open-vocab — distinguishes
    from 'computed but empty' which would be ()."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        install_extractor_plan,
    )

    plan = install_extractor_plan("llm")
    assert plan.canonicalization_candidates is None
