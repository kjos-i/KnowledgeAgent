"""Tests for `embedder_lifecycle` — install / uninstall / HF model
download / dimension-change guard.

Mirrors test_llm_lifecycle for the embedder side, plus the
dimension-change guard which is unique to embedders (the LanceDB
chunks-table vector field pins its dimension at creation).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.embedder_lifecycle import (
    EMBEDDER_PROVIDER_REGISTRY,
    HF_EMBEDDING_MODELS,
    DownloadHFModelPlan,
    SwitchEmbedderPlan,
    UninstallEmbedderProviderPlan,
    download_hf_model_execute,
    download_hf_model_plan,
    install_embedder_provider_execute,
    install_embedder_provider_plan,
    switch_embedder_plan,
    uninstall_embedder_provider_execute,
    uninstall_embedder_provider_plan,
)


_PIP_PATCH = "knowledge_agent.embedder_lifecycle._run_pip"


# ---- registry sanity ----


def test_registry_has_four_providers():
    assert set(EMBEDDER_PROVIDER_REGISTRY) == {
        "voyage", "openai", "google", "huggingface"
    }


def test_no_provider_is_bundled_after_2026_06_29_refactor():
    """Pre-2026-06-29 Voyage was bundled (voyageai in base deps).
    The bundled-defaults removal moved it to the `embed-voyage`
    extra so all four embedder providers are symmetric — every one
    is opt-in via the first-launch wizard.
    """
    for name in ("voyage", "openai", "google", "huggingface"):
        entry = EMBEDDER_PROVIDER_REGISTRY[name]
        assert entry["bundled"] is False, name
        assert entry["provenance"].pip_extras is not None, name
        assert entry["library_packages"], name


def test_voyage_default_dim_is_1024():
    entry = EMBEDDER_PROVIDER_REGISTRY["voyage"]
    assert entry["provenance"].default_dim == 1024


def test_each_provider_provenance_has_default_dim():
    """Default dims drive the dim-change guard, so the assertion
    that ALL providers expose this is load-bearing."""
    expected = {
        "voyage": 1024,
        "openai": 1536,
        "google": 768,
        "huggingface": 1024,
    }
    for name, want_dim in expected.items():
        prov = EMBEDDER_PROVIDER_REGISTRY[name]["provenance"]
        assert prov.default_dim == want_dim


def test_hf_curated_menu_locked_to_four_entries():
    assert set(HF_EMBEDDING_MODELS) == {
        "BAAI/bge-m3",
        "mixedbread-ai/mxbai-embed-large-v1",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_all_hf_models_are_safetensors_with_pinned_revision():
    """No pickle-format models on the curated menu; pinned commit
    SHAs must NEVER be `main`."""
    for model_id, prov in HF_EMBEDDING_MODELS.items():
        assert prov.safetensors is True, model_id
        assert prov.pinned_revision != "main", model_id
        assert len(prov.pinned_revision) >= 12, model_id


# ---- install plan ----


def test_plan_voyage_surfaces_pip_extras_when_not_installed():
    """Post-bundled-defaults-removal: voyage is a provider like the
    others. When the adapter isn't installed, the plan summary
    surfaces the `embed-voyage` extra."""
    with patch.dict(
        EMBEDDER_PROVIDER_REGISTRY["voyage"],
        {"is_installed_fn": lambda: False},
    ):
        plan = install_embedder_provider_plan("voyage")
    assert plan.bundled is False
    assert plan.already_installed is False
    assert "embed-voyage" in plan.summary


def test_plan_huggingface_surfaces_disk_warning():
    """HF is the heaviest install — the summary must warn about
    sentence-transformers + torch (2-3 GB)."""
    with patch.dict(
        EMBEDDER_PROVIDER_REGISTRY["huggingface"],
        {"is_installed_fn": lambda: False},
    ):
        plan = install_embedder_provider_plan("huggingface")
    assert plan.already_installed is False
    assert "2-3 GB" in plan.summary
    assert "embed-huggingface" in plan.summary


def test_plan_openai_lists_dimension():
    """Dimension is surfaced in install summary so users see what
    they're committing to — required by the dim-change guard
    contract."""
    with patch.dict(
        EMBEDDER_PROVIDER_REGISTRY["openai"],
        {"is_installed_fn": lambda: False},
    ):
        plan = install_embedder_provider_plan("openai")
    assert "1536" in plan.summary


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        install_embedder_provider_plan("not-a-provider")


# ---- install execute ----


@pytest.mark.asyncio
async def test_execute_runs_pip_with_correct_extra():
    with patch.dict(
        EMBEDDER_PROVIDER_REGISTRY["openai"],
        {"is_installed_fn": lambda: False},
    ):
        plan = install_embedder_provider_plan("openai")
    with patch(
        _PIP_PATCH, new_callable=AsyncMock, return_value=(True, "ok"),
    ) as pip:
        result = await install_embedder_provider_execute(plan)
    args = pip.call_args.args[0]
    assert "[embed-openai]" in args[1]
    assert result.install_ok is True


# ---- uninstall blocking when active ----


def test_uninstall_blocked_when_provider_is_active():
    with (
        patch.dict(
            EMBEDDER_PROVIDER_REGISTRY["openai"],
            {"is_installed_fn": lambda: True},
        ),
        patch(
            "knowledge_agent.embedder_lifecycle.get_settings",
            return_value=MagicMock(embedding_provider="openai"),
        ),
    ):
        plan = uninstall_embedder_provider_plan("openai")
    assert plan.is_active is True
    assert "ACTIVE embedding provider" in plan.summary


@pytest.mark.asyncio
async def test_uninstall_execute_no_op_when_active():
    plan = UninstallEmbedderProviderPlan(
        provider_name="huggingface",
        display_name="HuggingFace (local)",
        packages_to_remove=("sentence-transformers", "torch"),
        installed=True,
        bundled=False,
        is_active=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip:
        result = await uninstall_embedder_provider_execute(plan)
    pip.assert_not_called()
    assert result.did_uninstall is False


@pytest.mark.asyncio
async def test_uninstall_runs_pip_when_inactive_and_installed():
    plan = UninstallEmbedderProviderPlan(
        provider_name="google",
        display_name="Google",
        packages_to_remove=("langchain-google-genai",),
        installed=True,
        bundled=False,
        is_active=False,
    )
    with patch(
        _PIP_PATCH, new_callable=AsyncMock, return_value=(True, "ok"),
    ) as pip:
        result = await uninstall_embedder_provider_execute(plan)
    pip.assert_called_once()
    assert pip.call_args.args[0] == [
        "uninstall", "-y", "langchain-google-genai",
    ]
    assert result.uninstall_ok is True


# ---- HF model download ----


def test_hf_download_plan_blocked_when_libs_not_installed():
    """libs (sentence-transformers + torch) must be installed first;
    the plan surfaces this so the GUI chains the two steps."""
    with patch(
        "knowledge_agent.embedder_lifecycle._hf_embed_libs_installed",
        return_value=False,
    ):
        plan = download_hf_model_plan("BAAI/bge-m3")
    assert plan.libs_installed is False
    assert "Install the HuggingFace libs first" in plan.summary


def test_hf_download_plan_when_libs_present():
    # download_hf_model_plan checks `_hf_embed_libs_installed()`
    # directly (not via registry), so patch that function.
    with patch(
        "knowledge_agent.embedder_lifecycle._hf_embed_libs_installed",
        return_value=True,
    ):
        plan = download_hf_model_plan("BAAI/bge-m3")
    assert plan.libs_installed is True
    assert "BAAI/bge-m3" in plan.provenance.model_id
    assert "2300" in plan.summary  # download_size_mb
    assert "Multilingual" in plan.summary


def test_hf_download_execute_short_circuits_when_libs_missing():
    plan = DownloadHFModelPlan(
        model_id="BAAI/bge-m3",
        provenance=HF_EMBEDDING_MODELS["BAAI/bge-m3"],
        libs_installed=False,
    )
    result = download_hf_model_execute(plan)
    assert result.did_download is False
    assert result.download_ok is False


def test_hf_download_execute_calls_snapshot_download_with_pinned_sha():
    plan = DownloadHFModelPlan(
        model_id="BAAI/bge-m3",
        provenance=HF_EMBEDDING_MODELS["BAAI/bge-m3"],
        libs_installed=True,
    )
    fake_snapshot = MagicMock()
    with patch.dict(
        "sys.modules",
        {"huggingface_hub": MagicMock(snapshot_download=fake_snapshot)},
    ):
        result = download_hf_model_execute(plan)
    fake_snapshot.assert_called_once_with(
        repo_id="BAAI/bge-m3",
        revision=HF_EMBEDDING_MODELS["BAAI/bge-m3"].pinned_revision,
    )
    assert result.download_ok is True


def test_unknown_hf_model_raises():
    with pytest.raises(ValueError, match="Curated menu"):
        download_hf_model_plan("not/a-real-model")


# ---- dimension-change guard ----


def test_switch_plan_no_data_means_safe_switch():
    """When the LanceDB chunks table doesn't exist or is empty,
    switching is always safe — no rows to wipe."""
    fake_client = MagicMock()
    fake_client.conn.table_names.return_value = []  # no tables
    with (
        patch(
            "knowledge_agent.embedder_lifecycle.get_settings",
            return_value=MagicMock(embedding_provider="voyage"),
        ),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_client,
        ),
    ):
        plan = switch_embedder_plan("openai")
    assert plan.from_provider == "voyage"
    assert plan.to_provider == "openai"
    assert plan.existing_rows == 0
    assert plan.dim_mismatch is False
    assert "No data yet" in plan.summary


def test_switch_plan_destructive_when_dims_differ_with_existing_rows():
    """voyage (1024) → google (768) with existing rows = destructive."""
    fake_table = MagicMock()
    fake_table.count_rows.return_value = 1234
    fake_field = MagicMock()
    fake_field.type.list_size = 1024
    fake_table.schema.field.return_value = fake_field

    fake_client = MagicMock()
    fake_client.conn.table_names.return_value = ["chunks"]
    fake_client.conn.open_table.return_value = fake_table

    with (
        patch(
            "knowledge_agent.embedder_lifecycle.get_settings",
            return_value=MagicMock(embedding_provider="voyage"),
        ),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_client,
        ),
    ):
        plan = switch_embedder_plan("google")

    assert plan.from_dim == 1024
    assert plan.to_dim == 768
    assert plan.existing_rows == 1234
    assert plan.dim_mismatch is True
    assert "DESTRUCTIVE" in plan.summary
    assert "1234" in plan.summary


def test_switch_plan_same_dim_keeps_data_no_destructive_flag():
    """voyage (1024) → huggingface (1024 default) — dims align, no
    destructive flag; just a softer "consider re-embedding" hint."""
    fake_table = MagicMock()
    fake_table.count_rows.return_value = 500
    fake_field = MagicMock()
    fake_field.type.list_size = 1024
    fake_table.schema.field.return_value = fake_field

    fake_client = MagicMock()
    fake_client.conn.table_names.return_value = ["chunks"]
    fake_client.conn.open_table.return_value = fake_table

    with (
        patch(
            "knowledge_agent.embedder_lifecycle.get_settings",
            return_value=MagicMock(embedding_provider="voyage"),
        ),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_client,
        ),
    ):
        plan = switch_embedder_plan("huggingface")

    assert plan.dim_mismatch is False
    assert plan.existing_rows == 500
    assert "DESTRUCTIVE" not in plan.summary


def test_switch_plan_same_provider_summary_says_no_change():
    fake_client = MagicMock()
    fake_client.conn.table_names.return_value = []
    with (
        patch(
            "knowledge_agent.embedder_lifecycle.get_settings",
            return_value=MagicMock(embedding_provider="voyage"),
        ),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_client,
        ),
    ):
        plan = switch_embedder_plan("voyage")
    assert "no change" in plan.summary.lower()
