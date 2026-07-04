"""Tests for `llm_lifecycle` — install / uninstall / Ollama model pull.

Covers:
  - Registry sanity: 4 providers with the right shape.
  - Install plan: bundled (Anthropic), not-installed (OpenAI), with-
    provenance summary content.
  - Uninstall plan: BLOCKED when the provider is the active
    `settings.llm_provider`.
  - Ollama daemon detection (PATH + HTTP fallback).
  - `pull_ollama_model_*` ops gated on daemon reachability.

No real pip / ollama / network calls — `_run_pip`, `_run_ollama`,
`shutil.which`, and `httpx.get` are patched.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_http_get(*, status_code: int | None = None, exc: Exception | None = None):
    """Build an `AsyncMock` for `_http_client.get` that returns a response
    with `status_code` OR raises `exc`. Use as `patch("..._http_client.get", _fake_http_get(...))`.
    """
    fake_resp = MagicMock(status_code=status_code or 0)
    return AsyncMock(
        side_effect=exc if exc is not None else None,
        return_value=fake_resp if exc is None else None,
    )


_HTTP_PATCH = "knowledge_agent._http_client.request"

from knowledge_agent.llm_lifecycle import (
    LLM_PROVIDER_REGISTRY,
    OLLAMA_MODELS,
    PullOllamaModelPlan,
    UninstallLLMProviderPlan,
    _ollama_daemon_is_reachable,
    install_llm_provider_execute,
    install_llm_provider_plan,
    pull_ollama_model_execute,
    pull_ollama_model_plan,
    uninstall_llm_provider_execute,
    uninstall_llm_provider_plan,
)

_PIP_PATCH = "knowledge_agent.llm_lifecycle._run_pip"
_OLLAMA_PATCH = "knowledge_agent.llm_lifecycle._run_ollama"
_WHICH_PATCH = "knowledge_agent.llm_lifecycle.shutil.which"


# ---- registry sanity ----


def test_registry_has_four_providers():
    assert set(LLM_PROVIDER_REGISTRY) == {"anthropic", "openai", "google", "ollama"}


def test_no_provider_is_bundled_after_2026_06_29_refactor():
    """Pre-2026-06-29 Anthropic was bundled (in base deps). The
    bundled-defaults removal moved langchain-anthropic to the
    `llm-anthropic` extra so all four providers are symmetric:
    every one is opt-in via the first-launch wizard.
    """
    for name in ("anthropic", "openai", "google", "ollama"):
        entry = LLM_PROVIDER_REGISTRY[name]
        assert entry["bundled"] is False, name
        assert entry["provenance"].pip_extras is not None, name
        assert entry["library_packages"], name


def test_each_provider_has_a_pip_extras_name():
    expected = {
        "anthropic": "llm-anthropic",
        "openai": "llm-openai",
        "google": "llm-google",
        "ollama": "llm-ollama",
    }
    for name, extras in expected.items():
        assert LLM_PROVIDER_REGISTRY[name]["provenance"].pip_extras == extras


def test_ollama_provenance_flags_daemon_requirement():
    entry = LLM_PROVIDER_REGISTRY["ollama"]
    prov = entry["provenance"]
    assert prov.requires_daemon is True
    assert prov.daemon_install_url == "https://ollama.com/download"


def test_each_provider_has_six_default_model_entries():
    """One model per call site (mode_classifier / query_builder /
    cypher_builder / synthesizer / entity_extractor /
    triples_extractor) keeps the lifecycle switch step able to
    rewrite all six Settings fields in one go."""
    expected_sites = {
        "mode_classifier",
        "query_builder",
        "cypher_builder",
        "synthesizer",
        "entity_extractor",
        "triples_extractor",
    }
    for name, entry in LLM_PROVIDER_REGISTRY.items():
        assert set(entry["default_models"]) == expected_sites, name


def test_ollama_curated_menu_locked_to_four_entries():
    assert set(OLLAMA_MODELS) == {
        "llama3.2:3b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "llama3.3:70b",
    }


# ---- install_llm_provider_plan ----


@pytest.mark.asyncio
async def test_plan_anthropic_surfaces_pip_extras_when_not_installed():
    """Post-bundled-defaults-removal: anthropic is a provider like
    the others. When the adapter isn't installed, the plan summary
    surfaces the `llm-anthropic` extra so the wizard / GUI can
    install it."""
    with patch.dict(
        LLM_PROVIDER_REGISTRY["anthropic"],
        {"is_installed_fn": lambda: False},
    ):
        plan = await install_llm_provider_plan("anthropic")
    assert plan.bundled is False
    assert plan.already_installed is False
    assert "llm-anthropic" in plan.summary


@pytest.mark.asyncio
async def test_plan_openai_when_not_installed_surfaces_pip_extras():
    with patch.dict(
        LLM_PROVIDER_REGISTRY["openai"],
        {"is_installed_fn": lambda: False},
    ):
        plan = await install_llm_provider_plan("openai")
    assert plan.bundled is False
    assert plan.already_installed is False
    assert "llm-openai" in plan.summary
    assert "OpenAI" in plan.summary


@pytest.mark.asyncio
async def test_plan_ollama_when_daemon_missing_surfaces_install_url():
    """Daemon-not-reachable adds the manual install URL to the
    summary so the GUI can guide the user."""
    with (
        patch.dict(
            LLM_PROVIDER_REGISTRY["ollama"],
            {"is_installed_fn": lambda: False},
        ),
        patch(_WHICH_PATCH, return_value=None),
        # Force HTTP fallback to fail too.
        patch(
            "knowledge_agent.llm_lifecycle.get_settings",
            return_value=MagicMock(ollama_base_url="http://localhost:11434"),
        ),
        patch(
            _HTTP_PATCH,
            _fake_http_get(exc=ConnectionError("daemon down")),
        ),
    ):
        plan = await install_llm_provider_plan("ollama")
    assert plan.daemon_reachable is False
    assert "https://ollama.com/download" in plan.summary


@pytest.mark.asyncio
async def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        await install_llm_provider_plan("not-a-provider")


# ---- install_llm_provider_execute ----


@pytest.mark.asyncio
async def test_execute_no_op_when_already_installed():
    """No provider is bundled post-2026-06-29, but already-installed
    is still a valid no-op state (e.g. user installs anthropic via
    wizard, then a later install_llm_provider_execute call shouldn't
    re-pip).
    """
    with patch.dict(
        LLM_PROVIDER_REGISTRY["anthropic"],
        {"is_installed_fn": lambda: True},
    ):
        plan = await install_llm_provider_plan("anthropic")
    result = await install_llm_provider_execute(plan)
    assert result.did_install is False
    assert result.install_ok is True


@pytest.mark.asyncio
async def test_execute_runs_pip_when_not_installed():
    with patch.dict(
        LLM_PROVIDER_REGISTRY["openai"],
        {"is_installed_fn": lambda: False},
    ):
        plan = await install_llm_provider_plan("openai")
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "install ok"),
    ) as pip:
        result = await install_llm_provider_execute(plan)
    pip.assert_called_once()
    args = pip.call_args.args[0]
    assert args[0] == "install"
    assert "[llm-openai]" in args[1]
    assert result.install_ok is True
    assert result.restart_required is True


# ---- uninstall blocking when active ----


def test_uninstall_blocked_when_provider_is_active():
    with (
        patch.dict(
            LLM_PROVIDER_REGISTRY["openai"],
            {"is_installed_fn": lambda: True},
        ),
        patch(
            "knowledge_agent.llm_lifecycle.get_settings",
            return_value=MagicMock(llm_provider="openai"),
        ),
    ):
        plan = uninstall_llm_provider_plan("openai")
    assert plan.is_active is True
    assert "ACTIVE LLM provider" in plan.summary
    assert "Switch to a different provider" in plan.summary


@pytest.mark.asyncio
async def test_uninstall_execute_no_op_when_active():
    plan = UninstallLLMProviderPlan(
        provider_name="openai",
        display_name="OpenAI GPT",
        packages_to_remove=("langchain-openai",),
        installed=True,
        bundled=False,
        is_active=True,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip:
        result = await uninstall_llm_provider_execute(plan)
    pip.assert_not_called()
    assert result.did_uninstall is False
    assert result.uninstall_ok is True


@pytest.mark.asyncio
async def test_uninstall_runs_pip_when_inactive_and_installed():
    plan = UninstallLLMProviderPlan(
        provider_name="google",
        display_name="Google Gemini",
        packages_to_remove=("langchain-google-genai",),
        installed=True,
        bundled=False,
        is_active=False,
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "uninstall ok"),
    ) as pip:
        result = await uninstall_llm_provider_execute(plan)
    pip.assert_called_once()
    assert pip.call_args.args[0] == [
        "uninstall",
        "-y",
        "langchain-google-genai",
    ]
    assert result.uninstall_ok is True


# ---- ollama daemon detection ----


@pytest.mark.asyncio
async def test_daemon_reachable_when_binary_on_path():
    with patch(_WHICH_PATCH, return_value="/usr/local/bin/ollama"):
        assert await _ollama_daemon_is_reachable() is True


@pytest.mark.asyncio
async def test_daemon_reachable_via_http_when_binary_not_on_path():
    with (
        patch(_WHICH_PATCH, return_value=None),
        patch(
            "knowledge_agent.llm_lifecycle.get_settings",
            return_value=MagicMock(ollama_base_url="http://localhost:11434"),
        ),
        patch(
            _HTTP_PATCH,
            _fake_http_get(status_code=200),
        ),
    ):
        assert await _ollama_daemon_is_reachable() is True


@pytest.mark.asyncio
async def test_daemon_not_reachable_when_neither_works():
    with (
        patch(_WHICH_PATCH, return_value=None),
        patch(
            "knowledge_agent.llm_lifecycle.get_settings",
            return_value=MagicMock(ollama_base_url="http://localhost:11434"),
        ),
        patch(
            _HTTP_PATCH,
            _fake_http_get(exc=ConnectionError("down")),
        ),
    ):
        assert await _ollama_daemon_is_reachable() is False


# ---- pull_ollama_model ----


@pytest.mark.asyncio
async def test_pull_plan_summary_when_daemon_reachable():
    with patch(_WHICH_PATCH, return_value="/usr/local/bin/ollama"):
        plan = await pull_ollama_model_plan("qwen2.5:7b")
    assert plan.daemon_reachable is True
    assert "qwen2.5:7b" in plan.summary
    assert "4.7" in plan.summary  # download_size_gb
    assert "Apache-2.0" in plan.summary


@pytest.mark.asyncio
async def test_pull_plan_summary_when_daemon_not_reachable():
    with (
        patch(_WHICH_PATCH, return_value=None),
        patch(
            "knowledge_agent.llm_lifecycle.get_settings",
            return_value=MagicMock(ollama_base_url="http://localhost:11434"),
        ),
        patch(
            _HTTP_PATCH,
            _fake_http_get(exc=ConnectionError("down")),
        ),
    ):
        plan = await pull_ollama_model_plan("llama3.2:3b")
    assert plan.daemon_reachable is False
    assert "Ollama daemon is not reachable" in plan.summary


@pytest.mark.asyncio
async def test_pull_execute_short_circuits_when_daemon_unreachable():
    plan = PullOllamaModelPlan(
        model_id="qwen2.5:7b",
        provenance=OLLAMA_MODELS["qwen2.5:7b"],
        daemon_reachable=False,
    )
    with patch(_OLLAMA_PATCH, new_callable=AsyncMock) as run:
        result = await pull_ollama_model_execute(plan)
    run.assert_not_called()
    assert result.did_pull is False
    assert result.pull_ok is False


@pytest.mark.asyncio
async def test_pull_execute_runs_ollama_pull_when_daemon_reachable():
    plan = PullOllamaModelPlan(
        model_id="qwen2.5:7b",
        provenance=OLLAMA_MODELS["qwen2.5:7b"],
        daemon_reachable=True,
    )
    with patch(
        _OLLAMA_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "pulled"),
    ) as run:
        result = await pull_ollama_model_execute(plan)
    run.assert_called_once_with(["pull", "qwen2.5:7b"])
    assert result.did_pull is True
    assert result.pull_ok is True


@pytest.mark.asyncio
async def test_unknown_ollama_model_raises():
    with pytest.raises(ValueError, match="Curated menu"):
        await pull_ollama_model_plan("not-a-curated-model")
