"""Tests for entity_extractors.extractor_lifecycle - install / cache / uninstall."""

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_agent import DISTRIBUTION_NAME
from knowledge_agent.entity_extractors.extractor_lifecycle import (
    EXTRACTOR_REGISTRY,
    DeleteExtractorCachePlan,
    DeleteExtractorCacheResult,
    InstallExtractorPlan,
    InstallExtractorResult,
    ModelProvenance,
    UninstallExtractorPlan,
    UninstallExtractorResult,
    _scan_pickle_weights,
    delete_extractor_cache_execute,
    delete_extractor_cache_plan,
    install_extractor_execute,
    install_extractor_plan,
    uninstall_extractor_execute,
    uninstall_extractor_plan,
)


@pytest.mark.security
def test_scan_pickle_weights_rejects_malicious_pickle(tmp_path):
    """A pickle carrying a code-exec __reduce__ is flagged before load (audit H)."""
    pytest.importorskip("picklescan")
    import os
    import pickle

    class _Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    (tmp_path / "pytorch_model.bin").write_bytes(pickle.dumps(_Evil()))
    assert _scan_pickle_weights(tmp_path) is not None


@pytest.mark.security
def test_scan_pickle_weights_passes_clean_pickle(tmp_path):
    """A plain-data pickle is not flagged (audit H)."""
    pytest.importorskip("picklescan")
    import pickle

    (tmp_path / "pytorch_model.bin").write_bytes(pickle.dumps({"weights": [1, 2, 3]}))
    assert _scan_pickle_weights(tmp_path) is None


def _sample_provenance(**overrides) -> ModelProvenance:
    """Build a ModelProvenance for tests with sensible safe defaults."""
    defaults = {
        "model_name": "test-org/test-model",
        "source_url": "https://huggingface.co/test-org/test-model",
        "download_size_mb": 500,
        "publisher": "Test Org (Test University)",
        "license": "Apache-2.0",
        "safetensors": True,
        "trust_remote_code": False,
        "pinned_revision": "0123456789abcdef0123456789abcdef01234567",
    }
    defaults.update(overrides)
    return ModelProvenance(**defaults)


_PIP_PATCH = "knowledge_agent.entity_extractors.extractor_lifecycle._run_pip"


# ---- registry ----


def test_registry_has_core_adapters():
    """LLM + GLiNER family + HunFlair2 all live in the registry."""
    assert set(EXTRACTOR_REGISTRY) >= {
        "llm",
        "gliner",
        "gliner_biomed",
        "hunflair2",
    }


def test_registry_llm_is_bundled_and_has_no_extras():
    entry = EXTRACTOR_REGISTRY["llm"]
    assert entry["bundled"] is True
    assert entry["pip_extras"] is None
    assert entry["model_packages"] == ()


def test_registry_llm_provenance_is_none_because_bundled():
    """Bundled adapters never download — no provenance to surface."""
    assert EXTRACTOR_REGISTRY["llm"]["provenance"] is None


def test_registry_scispacy_is_removed():
    """SciSpaCy was demoted 2026-06-23 when HunFlair2 shipped
    (HunFlair2 outperforms by ~9 pp F1). Verify it's no longer in
    the registry — catches accidental re-add."""
    assert "scispacy" not in EXTRACTOR_REGISTRY


def test_registry_has_hunflair2_entry():
    """HunFlair2 ships 2026-06-23, supersedes SciSpaCy."""
    assert "hunflair2" in EXTRACTOR_REGISTRY
    entry = EXTRACTOR_REGISTRY["hunflair2"]
    assert entry["bundled"] is False
    assert entry["pip_extras"] == "entities-hunflair2"


def test_registry_hunflair2_provenance_flags_no_safetensors():
    """The hunflair2-ner checkpoint ships only pytorch_model.bin —
    no .safetensors. Provenance MUST report safetensors=False so the
    install dialog flags the pickle exec risk."""
    entry = EXTRACTOR_REGISTRY["hunflair2"]
    prov = entry["provenance"]
    assert prov is not None
    assert isinstance(prov, ModelProvenance)
    assert prov.model_name == "hunflair/hunflair2-ner"
    assert len(prov.pinned_revision) == 40
    assert prov.pinned_revision != "main"
    assert prov.safetensors is False
    assert prov.trust_remote_code is False
    # MIT inherited from the Flair library
    assert "MIT" in prov.license


def test_hunflair2_adapter_revision_matches_registry_provenance():
    """Pinned SHA in the adapter must equal the SHA in the provenance
    (drift catcher; same shape as gliner)."""
    from knowledge_agent.entity_extractors import hunflair2

    entry = EXTRACTOR_REGISTRY["hunflair2"]
    assert entry["provenance"].model_name == hunflair2.MODEL_NAME
    assert entry["provenance"].pinned_revision == hunflair2.MODEL_REVISION


def test_registry_has_gliner_entry():
    """GLiNER ships 2026-06-23 with full provenance."""
    assert "gliner" in EXTRACTOR_REGISTRY
    entry = EXTRACTOR_REGISTRY["gliner"]
    assert entry["bundled"] is False
    assert entry["pip_extras"] == "entities-gliner"


def test_registry_gliner_provenance_has_pinned_revision():
    """GLiNER's provenance must carry a real 40-char commit SHA, never
    `main`. Catches accidental un-pin during config edits."""
    entry = EXTRACTOR_REGISTRY["gliner"]
    prov = entry["provenance"]
    assert prov is not None
    assert isinstance(prov, ModelProvenance)
    assert prov.model_name == "urchade/gliner_multi-v2.1"
    assert len(prov.pinned_revision) == 40
    assert prov.pinned_revision != "main"
    assert prov.safetensors is True
    assert prov.trust_remote_code is False
    assert prov.license == "Apache-2.0"


def test_gliner_adapter_revision_matches_registry_provenance():
    """The pinned revision in the adapter MUST match the provenance
    surfaced in the install dialog. They are two copies of the same
    truth — keep them in sync. Test catches drift if one is edited
    without the other."""
    from knowledge_agent.entity_extractors import gliner

    entry = EXTRACTOR_REGISTRY["gliner"]
    assert entry["provenance"].model_name == gliner.MODEL_NAME
    assert entry["provenance"].pinned_revision == gliner.MODEL_REVISION


def test_registry_has_gliner_biomed_entry():
    """GLiNER-BioMed ships 2026-06-23 with full provenance."""
    assert "gliner_biomed" in EXTRACTOR_REGISTRY
    entry = EXTRACTOR_REGISTRY["gliner_biomed"]
    assert entry["bundled"] is False
    assert entry["pip_extras"] == "entities-gliner-biomed"


def test_registry_gliner_biomed_provenance_explicitly_flags_no_safetensors():
    """The bi-large GLiNER-BioMed checkpoint ships only
    pytorch_model.bin — no .safetensors. The provenance MUST report
    safetensors=False so the install dialog flags the pickle exec risk."""
    entry = EXTRACTOR_REGISTRY["gliner_biomed"]
    prov = entry["provenance"]
    assert prov is not None
    assert isinstance(prov, ModelProvenance)
    assert prov.model_name == "Ihor/gliner-biomed-bi-large-v1.0"
    assert len(prov.pinned_revision) == 40
    assert prov.pinned_revision != "main"
    # Critical: bi-large variant has no safetensors file as of
    # 2026-06-23. Confirm honest disclosure.
    assert prov.safetensors is False
    assert prov.trust_remote_code is False
    assert prov.license == "Apache-2.0"


def test_gliner_biomed_adapter_revision_matches_registry_provenance():
    """Same drift-catcher as the general gliner adapter: pinned SHA
    in the adapter code must equal the SHA in the provenance."""
    from knowledge_agent.entity_extractors import gliner_biomed

    entry = EXTRACTOR_REGISTRY["gliner_biomed"]
    assert entry["provenance"].model_name == gliner_biomed.MODEL_NAME
    assert entry["provenance"].pinned_revision == gliner_biomed.MODEL_REVISION


# ---- ModelProvenance ----


def test_model_provenance_carries_security_disclosure_fields():
    """All eight fields the GUI dialog needs are present + typed."""
    prov = _sample_provenance()
    assert prov.model_name == "test-org/test-model"
    assert prov.source_url.startswith("https://")
    assert isinstance(prov.download_size_mb, int)
    assert prov.publisher
    assert prov.license
    assert prov.safetensors is True
    assert prov.trust_remote_code is False
    # 40-char SHA — never `main` or a branch name.
    assert len(prov.pinned_revision) == 40


def test_model_provenance_is_frozen():
    """Immutable so the same provenance can be reused across plans
    without aliasing surprises."""
    prov = _sample_provenance()
    with pytest.raises(Exception):
        prov.safetensors = False  # type: ignore[misc]


# ---- install_extractor_plan / execute ----


def test_install_plan_raises_on_unknown_extractor():
    with pytest.raises(ValueError):
        install_extractor_plan("not-a-real-extractor")


def test_install_plan_for_bundled_llm_marks_already_available():
    plan = install_extractor_plan("llm")
    assert plan.bundled is True
    assert plan.already_installed is True
    assert "ships by default" in plan.summary


def test_install_plan_for_not_installed_extractor_mentions_pip_command():
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: False,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = install_extractor_plan("hunflair2")

    assert plan.already_installed is False
    assert "entities-hunflair2" in plan.summary
    assert "restart" in plan.summary.lower()


def test_install_plan_for_already_installed_extractor_short_circuits():
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: True,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = install_extractor_plan("hunflair2")

    assert plan.already_installed is True
    assert "already installed" in plan.summary


def test_install_plan_threads_provenance_from_registry():
    """When the registry entry carries provenance, the plan reflects it
    so the GUI can show the security-disclosure dialog."""
    prov = _sample_provenance()
    fake_registry = {
        "fake-adapter": {
            "display_name": "Fake Adapter",
            "bundled": False,
            "pip_extras": "entities-fake",
            "model_packages": (),
            "is_installed_fn": lambda: False,
            "provenance": prov,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = install_extractor_plan("fake-adapter")

    assert plan.provenance is prov


def test_install_plan_provenance_is_none_when_registry_omits_it():
    """Backward compat: a registry entry without `provenance` key still
    produces a valid plan (uses dict.get → None)."""
    fake_registry = {
        "legacy-adapter": {
            "display_name": "Legacy",
            "bundled": False,
            "pip_extras": "entities-legacy",
            "model_packages": (),
            "is_installed_fn": lambda: False,
            # No "provenance" key at all - mirrors the pre-2026-06-23 shape.
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = install_extractor_plan("legacy-adapter")

    assert plan.provenance is None


def test_install_plan_summary_without_provenance_is_terse():
    """No provenance → short summary (existing behaviour preserved)."""
    plan = InstallExtractorPlan(
        extractor_name="legacy",
        display_name="Legacy",
        bundled=False,
        pip_extras="entities-legacy",
        already_installed=False,
        provenance=None,
    )
    assert "Downloads" not in plan.summary
    assert "entities-legacy" in plan.summary


def test_install_plan_summary_with_provenance_surfaces_security_fields():
    """When provenance present, summary names model, size, publisher,
    license, source URL, pinned revision (first 12 chars), and the
    safetensors + trust_remote_code flags so even text-only callers
    (CLI/smoke) see the disclosure."""
    prov = _sample_provenance(
        model_name="urchade/gliner_multi-v2.1",
        source_url="https://huggingface.co/urchade/gliner_multi-v2.1",
        download_size_mb=1200,
        publisher="Urchade Zaratiana (Sorbonne)",
        license="Apache-2.0",
        safetensors=True,
        trust_remote_code=False,
        pinned_revision="abcdef0123456789abcdef0123456789abcdef01",
    )
    plan = InstallExtractorPlan(
        extractor_name="gliner",
        display_name="GLiNER",
        bundled=False,
        pip_extras="entities-gliner",
        already_installed=False,
        provenance=prov,
    )
    summary = plan.summary
    assert "urchade/gliner_multi-v2.1" in summary
    assert "1200 MB" in summary
    assert "Urchade Zaratiana" in summary
    assert "Apache-2.0" in summary
    assert "huggingface.co/urchade" in summary
    assert "abcdef012345" in summary  # first 12 chars of pinned SHA
    assert "safetensors=True" in summary
    assert "trust_remote_code=False" in summary


def test_install_plan_summary_with_trust_remote_code_true_surfaces_it():
    """trust_remote_code=True is the highest-severity disclosure;
    confirm it lands as the raw flag AND a prominent SECURITY warning
    explaining the load-time-code risk."""
    prov = _sample_provenance(trust_remote_code=True)
    plan = InstallExtractorPlan(
        extractor_name="x",
        display_name="X",
        bundled=False,
        pip_extras="entities-x",
        already_installed=False,
        provenance=prov,
    )
    assert "trust_remote_code=True" in plan.summary
    # Per the 0d security review, the dialog appends an explicit
    # SECURITY warning when this flag is True.
    assert "SECURITY:" in plan.summary
    assert "load time" in plan.summary.lower()


def test_install_plan_summary_with_pickle_format_surfaces_it():
    """safetensors=False means model loads via pickle — dialog must
    surface the raw flag AND a prominent SECURITY warning explaining
    why (arbitrary code execution at load time)."""
    prov = _sample_provenance(safetensors=False)
    plan = InstallExtractorPlan(
        extractor_name="x",
        display_name="X",
        bundled=False,
        pip_extras="entities-x",
        already_installed=False,
        provenance=prov,
    )
    assert "safetensors=False" in plan.summary
    # The 0d security review (2026-06-30) added an explicit warning
    # appended via `_provenance.security_warning_text`; tighten the
    # assertion so a regression that drops it fails here.
    assert "SECURITY:" in plan.summary
    assert "pickle" in plan.summary.lower()


@pytest.mark.asyncio
async def test_install_execute_bundled_is_noop_does_not_run_pip():
    plan = InstallExtractorPlan(
        extractor_name="llm",
        display_name="LLM",
        bundled=True,
        pip_extras=None,
        already_installed=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await install_extractor_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_install is False
    assert result.install_ok is True
    assert result.restart_required is False


@pytest.mark.asyncio
async def test_install_execute_already_installed_is_noop():
    plan = InstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        already_installed=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await install_extractor_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_install is False


@pytest.mark.asyncio
async def test_install_execute_runs_pip_with_distribution_and_extras():
    plan = InstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        already_installed=False,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    ) as pip_mock:
        result = await install_extractor_execute(plan)

    args, _ = pip_mock.call_args
    # Default distribution_name -> the shared DISTRIBUTION_NAME (the pip name
    # the app was installed under), NOT the pre-rename literal that made every
    # extractor install fail.
    assert args[0] == [
        "install",
        f"{DISTRIBUTION_NAME}[entities-hunflair2]",
    ]
    assert result.did_install is True
    assert result.install_ok is True
    assert result.restart_required is True


@pytest.mark.asyncio
async def test_install_execute_propagates_pip_failure():
    plan = InstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        already_installed=False,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(False, "ERROR: no permission"),
    ):
        result = await install_extractor_execute(plan)

    assert result.did_install is True
    assert result.install_ok is False
    # Don't suggest a restart for a failed install.
    assert result.restart_required is False
    assert "ERROR" in result.pip_output


@pytest.mark.asyncio
async def test_install_execute_returns_result_dataclass():
    plan = InstallExtractorPlan(
        extractor_name="llm",
        display_name="LLM",
        bundled=True,
        pip_extras=None,
        already_installed=True,
    )
    assert isinstance(
        await install_extractor_execute(plan),
        InstallExtractorResult,
    )


# ---- delete_extractor_cache_plan / execute ----


def test_delete_cache_plan_raises_on_unknown_extractor():
    with pytest.raises(ValueError):
        delete_extractor_cache_plan("not-real")


def test_delete_cache_plan_for_llm_says_nothing_to_delete():
    plan = delete_extractor_cache_plan("llm")
    assert plan.model_packages == ()
    assert plan.installed is True  # bundled, always installed
    assert "no separate model cache" in plan.summary


def test_delete_cache_plan_for_not_installed_extractor():
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: False,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = delete_extractor_cache_plan("hunflair2")

    assert plan.installed is False
    assert "not installed" in plan.summary.lower()


def test_delete_cache_plan_for_installed_extractor_lists_packages():
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: True,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = delete_extractor_cache_plan("hunflair2")

    assert plan.installed is True
    assert plan.model_packages == ("flair-model",)
    assert "flair-model" in plan.summary


@pytest.mark.asyncio
async def test_delete_cache_execute_noop_when_no_model_packages():
    plan = DeleteExtractorCachePlan(
        extractor_name="llm",
        display_name="LLM",
        model_packages=(),
        installed=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await delete_extractor_cache_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_delete is False
    assert result.delete_ok is True


@pytest.mark.asyncio
async def test_delete_cache_execute_noop_when_not_installed():
    plan = DeleteExtractorCachePlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        model_packages=("flair-model",),
        installed=False,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await delete_extractor_cache_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_delete is False


@pytest.mark.asyncio
async def test_delete_cache_execute_runs_pip_uninstall_with_y_flag():
    plan = DeleteExtractorCachePlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        model_packages=("flair-model",),
        installed=True,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "removed"),
    ) as pip_mock:
        result = await delete_extractor_cache_execute(plan)

    args, _ = pip_mock.call_args
    # `-y` so pip doesn't try to interactively confirm.
    assert args[0] == ["uninstall", "-y", "flair-model"]
    assert result.did_delete is True
    assert result.delete_ok is True


@pytest.mark.asyncio
async def test_delete_cache_execute_propagates_failure():
    plan = DeleteExtractorCachePlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        model_packages=("flair-model",),
        installed=True,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(False, "permission denied"),
    ):
        result = await delete_extractor_cache_execute(plan)

    assert result.delete_ok is False


@pytest.mark.asyncio
async def test_delete_cache_execute_returns_result_dataclass():
    plan = DeleteExtractorCachePlan(
        extractor_name="llm",
        display_name="LLM",
        model_packages=(),
        installed=True,
    )
    assert isinstance(
        await delete_extractor_cache_execute(plan),
        DeleteExtractorCacheResult,
    )


# ---- uninstall_extractor_plan / execute ----


def test_uninstall_plan_raises_on_unknown_extractor():
    with pytest.raises(ValueError):
        uninstall_extractor_plan("not-real")


def test_uninstall_plan_for_bundled_llm_blocks_uninstall():
    plan = uninstall_extractor_plan("llm")
    assert plan.bundled is True
    assert "bundled" in plan.summary.lower()
    assert "cannot" in plan.summary.lower()


def test_uninstall_plan_for_installed_hunflair2_includes_library_and_model():
    """Plan packs hunflair2 (library) + fake-model into
    packages_to_remove so one pip call removes everything."""
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: True,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = uninstall_extractor_plan("hunflair2")

    assert "flair" in plan.packages_to_remove
    assert "flair-model" in plan.packages_to_remove
    assert plan.installed is True


def test_uninstall_plan_for_not_installed_extractor():
    fake_registry = {
        "hunflair2": {
            "display_name": "HunFlair2",
            "bundled": False,
            "pip_extras": "entities-hunflair2",
            "model_packages": ("flair-model",),
            "is_installed_fn": lambda: False,
        },
    }
    with patch(
        "knowledge_agent.entity_extractors.extractor_lifecycle.EXTRACTOR_REGISTRY",
        fake_registry,
    ):
        plan = uninstall_extractor_plan("hunflair2")

    assert plan.installed is False
    assert "not installed" in plan.summary.lower()


@pytest.mark.asyncio
async def test_uninstall_execute_bundled_does_not_run_pip_and_reports_blocked():
    plan = UninstallExtractorPlan(
        extractor_name="llm",
        display_name="LLM",
        bundled=True,
        pip_extras=None,
        packages_to_remove=(),
        installed=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await uninstall_extractor_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_uninstall is False
    # Bundled = blocked uninstall is NOT an "ok" result; surface so UI
    # can show the user why it didn't run.
    assert result.uninstall_ok is False


@pytest.mark.asyncio
async def test_uninstall_execute_not_installed_is_noop_success():
    plan = UninstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        packages_to_remove=("hunflair2", "flair-model"),
        installed=False,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await uninstall_extractor_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_uninstall is False
    assert result.uninstall_ok is True


@pytest.mark.asyncio
async def test_uninstall_execute_runs_pip_uninstall_for_all_packages():
    plan = UninstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        packages_to_remove=("hunflair2", "flair-model"),
        installed=True,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    ) as pip_mock:
        result = await uninstall_extractor_execute(plan)

    args, _ = pip_mock.call_args
    assert args[0] == [
        "uninstall",
        "-y",
        "hunflair2",
        "flair-model",
    ]
    assert result.did_uninstall is True
    assert result.uninstall_ok is True
    assert result.restart_required is True


@pytest.mark.asyncio
async def test_uninstall_execute_failed_pip_does_not_suggest_restart():
    plan = UninstallExtractorPlan(
        extractor_name="hunflair2",
        display_name="HunFlair2",
        bundled=False,
        pip_extras="entities-hunflair2",
        packages_to_remove=("hunflair2", "flair-model"),
        installed=True,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(False, "denied"),
    ):
        result = await uninstall_extractor_execute(plan)

    assert result.uninstall_ok is False
    assert result.restart_required is False


@pytest.mark.asyncio
async def test_uninstall_execute_returns_result_dataclass():
    plan = UninstallExtractorPlan(
        extractor_name="llm",
        display_name="LLM",
        bundled=True,
        pip_extras=None,
        packages_to_remove=(),
        installed=True,
    )
    assert isinstance(
        await uninstall_extractor_execute(plan),
        UninstallExtractorResult,
    )


# ---- ModelProvenance.domain_tags + EXTRACTOR_REGISTRY emitted_labels ----


def test_model_provenance_domain_tags_default_empty():
    """`domain_tags` defaults to () for back-compat with constants
    constructed without specifying it (test fixtures, transitional
    code)."""
    p = _sample_provenance()
    assert p.domain_tags == ()


def test_model_provenance_carries_domain_tags_when_set():
    p = _sample_provenance(domain_tags=("medicine", "biology"))
    assert p.domain_tags == ("medicine", "biology")


def test_gliner_provenance_domain_tags_general():
    """GLiNER general adapter declares the `"general"` taxonomy tag."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        _GLINER_PROVENANCE,
    )

    assert _GLINER_PROVENANCE.domain_tags == ("general",)


def test_gliner_biomed_provenance_domain_tags_biomedical():
    """GLiNER-BioMed spans the biomedical taxonomy slots."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        _GLINER_BIOMED_PROVENANCE,
    )

    tags = _GLINER_BIOMED_PROVENANCE.domain_tags
    assert "medicine" in tags
    assert "biology" in tags
    assert "chemistry" in tags
    assert "proteins" in tags


def test_hunflair2_provenance_domain_tags_narrower_than_gliner_biomed():
    """HunFlair2 emits 5 labels vs GLiNER-BioMed's 8 — narrower
    domain coverage too."""
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        _GLINER_BIOMED_PROVENANCE,
        _HUNFLAIR2_PROVENANCE,
    )

    h_tags = set(_HUNFLAIR2_PROVENANCE.domain_tags)
    g_tags = set(_GLINER_BIOMED_PROVENANCE.domain_tags)
    assert h_tags <= g_tags  # HunFlair2 tags are a subset


# ---- EXTRACTOR_REGISTRY emitted_labels + domain_tags ----


def test_registry_llm_emitted_labels_is_none():
    """Open-vocabulary marker — None tells the cross-link helper not
    to attempt static label matching."""
    assert EXTRACTOR_REGISTRY["llm"]["emitted_labels"] is None


def test_registry_llm_domain_tags_is_empty_tuple():
    """LLM is domain-agnostic; the empty tuple is the correct
    representation (distinct from None which would imply 'unset')."""
    assert EXTRACTOR_REGISTRY["llm"]["domain_tags"] == ()


def test_registry_gliner_emitted_labels_matches_module_defaults():
    """The registry's `emitted_labels` for gliner equals the
    adapter module's DEFAULT_LABELS constant — single source of truth."""
    from knowledge_agent.entity_extractors.gliner import (
        DEFAULT_LABELS,
    )

    assert EXTRACTOR_REGISTRY["gliner"]["emitted_labels"] == DEFAULT_LABELS


def test_registry_gliner_biomed_emitted_labels_matches_module_defaults():
    from knowledge_agent.entity_extractors.gliner_biomed import (
        DEFAULT_LABELS,
    )

    assert EXTRACTOR_REGISTRY["gliner_biomed"]["emitted_labels"] == DEFAULT_LABELS


def test_registry_hunflair2_emitted_labels_is_fixed_5_label_set():
    """HunFlair2 is all-or-nothing — the registry's emitted_labels
    must match the fixed 5-label vocabulary the adapter emits."""
    labels = EXTRACTOR_REGISTRY["hunflair2"]["emitted_labels"]
    assert labels == ("DISEASE", "CHEMICAL", "GENE", "SPECIES", "CELL_LINE")


def test_registry_emitted_labels_present_for_every_adapter():
    """Every registry entry must carry `emitted_labels` (None for
    open-vocab; tuple for fixed-vocab). Catches accidental drops
    when adding new adapters."""
    for name, entry in EXTRACTOR_REGISTRY.items():
        assert "emitted_labels" in entry, f"{name}: missing emitted_labels"


def test_registry_domain_tags_present_for_every_adapter():
    """Every entry must carry domain_tags (() for domain-agnostic)."""
    for name, entry in EXTRACTOR_REGISTRY.items():
        assert "domain_tags" in entry, f"{name}: missing domain_tags"


# ---- InstallExtractorPlan summary renders the new lines ----


def test_install_plan_llm_summary_says_open_vocabulary():
    """LLM plan's summary surfaces the open-vocab marker so the user
    knows there's no static label list to filter by."""
    plan = install_extractor_plan("llm")
    s = plan.summary
    assert "open vocabulary" in s


def test_install_plan_gliner_summary_lists_emitted_labels():
    """Non-bundled adapter plans surface the static label vocabulary
    in the summary."""
    plan = install_extractor_plan("gliner")
    s = plan.summary
    # General-NER labels appear.
    assert "PERSON" in s
    assert "ORGANIZATION" in s


def test_install_plan_gliner_biomed_summary_lists_domain_tags():
    """Biomedical adapter plan surfaces both domain tags AND labels."""
    plan = install_extractor_plan("gliner_biomed")
    s = plan.summary
    assert "medicine" in s
    assert "DISEASE" in s
    assert "CHEMICAL" in s


def test_install_plan_hunflair2_summary_lists_5_labels():
    plan = install_extractor_plan("hunflair2")
    s = plan.summary
    for lbl in ("DISEASE", "CHEMICAL", "GENE", "SPECIES", "CELL_LINE"):
        assert lbl in s


def test_install_plan_factory_threads_emitted_labels_from_registry():
    """The factory routes `entry["emitted_labels"]` onto the
    InstallExtractorPlan field — round-trip check."""
    plan = install_extractor_plan("gliner_biomed")
    from knowledge_agent.entity_extractors.gliner_biomed import (
        DEFAULT_LABELS,
    )

    assert plan.emitted_labels == DEFAULT_LABELS


def test_install_plan_factory_threads_domain_tags_from_registry():
    plan = install_extractor_plan("gliner")
    assert plan.domain_tags == ("general",)
