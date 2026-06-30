"""Tests for ingestion.parser_lifecycle - install + uninstall ops.

Mirrors the test_extractor_lifecycle pattern: patch `_run_pip` for
execute tests, patch `PARSER_LIFECYCLE_REGISTRY` + `shutil.which` for
plan tests with controlled environment states. No real pip is invoked.
"""

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_agent.ingestion.parser_lifecycle import (
    InstallParserExtraPlan,
    InstallParserExtraResult,
    PARSER_LIFECYCLE_REGISTRY,
    UninstallParserExtraPlan,
    UninstallParserExtraResult,
    _system_dep_hint,
    install_parser_extra_execute,
    install_parser_extra_plan,
    uninstall_parser_extra_execute,
    uninstall_parser_extra_plan,
)


_PIP_PATCH = (
    "knowledge_agent.ingestion.parser_lifecycle._run_pip"
)
_WHICH_PATCH = (
    "knowledge_agent.ingestion.parser_lifecycle.shutil.which"
)


def _fake_registry(
    extra_name: str,
    *,
    is_installed: bool,
    system_deps: tuple[str, ...] = (),
    pip_extras: str = "parsers-fake",
    library_packages: tuple[str, ...] = ("fake-pkg",),
    display_name: str = "Fake Extra",
    system_deps_hints: dict | None = None,
) -> dict:
    return {
        extra_name: {
            "display_name": display_name,
            "pip_extras": pip_extras,
            "library_packages": library_packages,
            "system_deps": system_deps,
            "system_deps_hints": system_deps_hints or {},
            "is_installed_fn": lambda: is_installed,
        }
    }


# ---- registry sanity ----


def test_registry_has_asr_and_code():
    assert set(PARSER_LIFECYCLE_REGISTRY) == {"asr", "code"}


def test_registry_asr_declares_no_system_deps_after_ffmpeg_bundling():
    """ffmpeg used to be a manual system install; it's now bundled
    via `imageio-ffmpeg` in the `parsers-asr` extra so the system_deps
    tuple is empty. The bundled binary is wired onto PATH at parser-
    module import time via `ensure_bundled_ffmpeg_on_path()`."""
    entry = PARSER_LIFECYCLE_REGISTRY["asr"]
    assert entry["pip_extras"] == "parsers-asr"
    assert entry["system_deps"] == ()
    assert entry["system_deps_hints"] == {}
    # The two library packages: whisper for ASR + imageio-ffmpeg for
    # the bundled ffmpeg binary.
    assert "openai-whisper" in entry["library_packages"]
    assert "imageio-ffmpeg" in entry["library_packages"]


def test_registry_code_has_no_system_deps():
    entry = PARSER_LIFECYCLE_REGISTRY["code"]
    assert entry["pip_extras"] == "parsers-code"
    assert entry["system_deps"] == ()
    assert "tree-sitter-language-pack" in entry["library_packages"]


# ---- install_parser_extra_plan ----


def test_install_plan_raises_on_unknown_extra():
    with pytest.raises(ValueError, match="Unknown parser extra"):
        install_parser_extra_plan("does-not-exist")


def test_install_plan_not_installed_no_system_deps_mentions_pip():
    fake = _fake_registry("code", is_installed=False, system_deps=())
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ):
        plan = install_parser_extra_plan("code")

    assert plan.already_installed is False
    assert plan.missing_system_deps == ()
    assert "pip install" in plan.summary
    assert "parsers-fake" in plan.summary
    assert "restart" in plan.summary.lower()


def test_install_plan_already_installed_no_system_deps_says_ready():
    fake = _fake_registry("code", is_installed=True)
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ):
        plan = install_parser_extra_plan("code")

    assert plan.already_installed is True
    assert "ready" in plan.summary.lower()
    assert "pip install" not in plan.summary


def test_install_plan_asr_with_ffmpeg_present(monkeypatch):
    fake = _fake_registry(
        "asr",
        is_installed=False,
        system_deps=("ffmpeg",),
        pip_extras="parsers-asr",
        system_deps_hints={"ffmpeg": {"Windows": "winget install ffmpeg"}},
    )
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ), patch(_WHICH_PATCH, return_value="/usr/bin/ffmpeg"):
        plan = install_parser_extra_plan("asr")

    assert plan.system_deps_status == {"ffmpeg": "/usr/bin/ffmpeg"}
    assert plan.missing_system_deps == ()
    # No ffmpeg hint should leak into the summary since ffmpeg is present
    assert "winget install ffmpeg" not in plan.summary
    assert "pip install" in plan.summary


def test_install_plan_asr_with_ffmpeg_missing_includes_install_hint():
    fake = _fake_registry(
        "asr",
        is_installed=False,
        system_deps=("ffmpeg",),
        pip_extras="parsers-asr",
        system_deps_hints={
            "ffmpeg": {
                "Windows": "winget install ffmpeg",
                "Darwin": "brew install ffmpeg",
                "Linux": "apt install ffmpeg",
            }
        },
    )
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ), patch(_WHICH_PATCH, return_value=None):
        plan = install_parser_extra_plan("asr")

    assert plan.missing_system_deps == ("ffmpeg",)
    assert "ffmpeg" in plan.summary
    # One of the per-OS hints should appear (whichever matches the
    # platform the tests run on)
    assert any(
        hint in plan.summary
        for hint in ["winget install ffmpeg", "brew install ffmpeg", "apt install ffmpeg"]
    )


def test_install_plan_already_installed_but_ffmpeg_missing_still_flags_it():
    fake = _fake_registry(
        "asr",
        is_installed=True,
        system_deps=("ffmpeg",),
        pip_extras="parsers-asr",
        system_deps_hints={"ffmpeg": {"Linux": "apt install ffmpeg"}},
    )
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ), patch(_WHICH_PATCH, return_value=None):
        plan = install_parser_extra_plan("asr")

    assert plan.already_installed is True
    assert plan.missing_system_deps == ("ffmpeg",)
    # Must NOT say "ready" - the user has work to do
    assert "ready" not in plan.summary.lower()
    assert "ffmpeg" in plan.summary


# ---- install_parser_extra_execute ----


@pytest.mark.asyncio
async def test_install_execute_already_installed_does_not_run_pip():
    plan = InstallParserExtraPlan(
        extra_name="code",
        display_name="Code Parser",
        pip_extras="parsers-code",
        already_installed=True,
        system_deps_status={},
        system_deps_hints={},
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await install_parser_extra_execute(plan)

    pip_mock.assert_not_called()
    assert result.did_install is False
    assert result.install_ok is True
    assert result.restart_required is False


@pytest.mark.asyncio
async def test_install_execute_not_installed_calls_pip_with_extras_target():
    plan = InstallParserExtraPlan(
        extra_name="code",
        display_name="Code Parser",
        pip_extras="parsers-code",
        already_installed=False,
        system_deps_status={},
        system_deps_hints={},
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(True, "Successfully installed"),
    ) as pip_mock:
        result = await install_parser_extra_execute(plan)

    pip_mock.assert_called_once()
    args = pip_mock.call_args[0][0]
    assert args[0] == "install"
    # target is "research-literature-agent[parsers-code]"
    assert "research-literature-agent[parsers-code]" in args[1]
    assert result.did_install is True
    assert result.install_ok is True
    assert result.restart_required is True


@pytest.mark.asyncio
async def test_install_execute_pip_failure_sets_install_ok_false_and_no_restart():
    plan = InstallParserExtraPlan(
        extra_name="code", display_name="Code Parser",
        pip_extras="parsers-code", already_installed=False,
        system_deps_status={}, system_deps_hints={},
    )
    with patch(
        _PIP_PATCH,
        new_callable=AsyncMock,
        return_value=(False, "ERROR: could not"),
    ):
        result = await install_parser_extra_execute(plan)

    assert result.did_install is True
    assert result.install_ok is False
    assert result.restart_required is False
    assert "ERROR" in result.pip_output


# ---- uninstall_parser_extra_plan ----


def test_uninstall_plan_raises_on_unknown_extra():
    with pytest.raises(ValueError, match="Unknown parser extra"):
        uninstall_parser_extra_plan("does-not-exist")


def test_uninstall_plan_not_installed_says_so():
    fake = _fake_registry("code", is_installed=False, library_packages=("foo",))
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ):
        plan = uninstall_parser_extra_plan("code")

    assert plan.installed is False
    assert "not installed" in plan.summary.lower()


def test_uninstall_plan_installed_mentions_packages_and_restart():
    fake = _fake_registry(
        "code",
        is_installed=True,
        library_packages=("tree-sitter-language-pack",),
    )
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.PARSER_LIFECYCLE_REGISTRY",
        fake,
    ):
        plan = uninstall_parser_extra_plan("code")

    assert plan.installed is True
    assert "tree-sitter-language-pack" in plan.summary
    assert "restart" in plan.summary.lower()


# ---- uninstall_parser_extra_execute ----


@pytest.mark.asyncio
async def test_uninstall_execute_not_installed_is_noop():
    plan = UninstallParserExtraPlan(
        extra_name="code", display_name="Code",
        packages_to_remove=("foo",), installed=False,
    )
    with patch(_PIP_PATCH, new_callable=AsyncMock) as pip_mock:
        result = await uninstall_parser_extra_execute(plan)
    pip_mock.assert_not_called()
    assert result.did_uninstall is False
    assert result.uninstall_ok is True


@pytest.mark.asyncio
async def test_uninstall_execute_installed_calls_pip_uninstall():
    plan = UninstallParserExtraPlan(
        extra_name="code", display_name="Code",
        packages_to_remove=("tree-sitter-language-pack",), installed=True,
    )
    with patch(
        _PIP_PATCH, new_callable=AsyncMock, return_value=(True, "Removed"),
    ) as pip_mock:
        result = await uninstall_parser_extra_execute(plan)

    pip_mock.assert_called_once()
    args = pip_mock.call_args[0][0]
    assert args[:2] == ["uninstall", "-y"]
    assert "tree-sitter-language-pack" in args
    assert result.did_uninstall is True
    assert result.uninstall_ok is True
    assert result.restart_required is True


# ---- _system_dep_hint OS dispatch ----


def test_system_dep_hint_picks_current_os(monkeypatch):
    hints = {
        "ffmpeg": {
            "Windows": "winget install ffmpeg",
            "Darwin": "brew install ffmpeg",
            "Linux": "apt install ffmpeg",
        }
    }
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.platform.system",
        return_value="Darwin",
    ):
        assert _system_dep_hint("ffmpeg", hints) == "brew install ffmpeg"


def test_system_dep_hint_falls_back_when_os_unknown():
    """Unknown OS -> shows all hints joined - better than silent miss."""
    hints = {
        "ffmpeg": {
            "Windows": "winget install ffmpeg",
            "Linux": "apt install ffmpeg",
        }
    }
    with patch(
        "knowledge_agent.ingestion.parser_lifecycle.platform.system",
        return_value="OS/2",
    ):
        result = _system_dep_hint("ffmpeg", hints)
    assert "winget install ffmpeg" in result
    assert "apt install ffmpeg" in result


def test_system_dep_hint_generic_when_no_hints_registered():
    result = _system_dep_hint("rare_binary", {})
    assert "package manager" in result.lower()
    assert "rare_binary" in result
