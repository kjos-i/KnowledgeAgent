"""GUI-facing parser lifecycle ops (Python-env + system-dep admin).

Same plan/execute UI contract as `kg/ontology_lifecycle.py` and
`entity_extractors/extractor_lifecycle.py`. Lives in `ingestion/`
because the per-parser lifecycle details (pip extras name, install
check, system binaries) belong with the parser-strategy package.

Two ops at launch:

  - `install_parser_extra` — `pip install rla[parsers-<name>]` via
    subprocess against the same Python interpreter the GUI is running
    under. The `asr` extra ships its own ffmpeg via the
    `imageio-ffmpeg` wheel — no system binary required, one install
    command covers everything. `ensure_bundled_ffmpeg_on_path()`
    prepends the bundled binary's directory to `PATH` at import time
    so docling / whisper find it when they subprocess to `ffmpeg`.

  - `uninstall_parser_extra` — full removal of the optional library
    packages. Symmetric with install.

NO `delete_cache` op at launch. The two parser caches are different
beasts and not yet worth managing per-knob from the GUI:

  - ASR cache = the Whisper Turbo model file, kept in HuggingFace's
    cache (~/.cache/huggingface/hub/...). Cleanly identifying which
    files belong to Whisper Turbo specifically would couple us to
    HuggingFace internals.
  - Code cache = tree-sitter grammars, which ship inside the
    `tree-sitter-language-pack` wheel itself; there's no separate
    cache directory to clean. Uninstalling the extra removes the
    grammars too.

If freeing the Whisper model without a full uninstall becomes a real
need later, the cache-clearing logic lands here as a third op.

`PARSER_LIFECYCLE_REGISTRY` declares per-parser metadata. Both entries
are non-bundled (the base package doesn't ship docling[asr] or
tree-sitter-language-pack).

NOTE: subprocess pip calls assume a writeable Python environment
(venv or per-user install). In a frozen / immutable distribution, the
install op would need to surface "this distribution is read-only;
re-install the app with the extra included" - deferred until we
actually ship a frozen distribution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---- per-parser-extra registry ----


def _asr_is_installed() -> bool:
    """True iff docling's ASR pipeline can actually run end-to-end.

    Two libraries must be present:
      - `whisper` — openai-whisper, ships the Whisper Turbo loader.
      - `imageio_ffmpeg` — bundles the ffmpeg binary that docling /
        whisper subprocess to for audio extraction.

    Both come in via the `parsers-asr` pip extra; the ASR pipeline
    needs both to function, so we report installed only when both are
    importable. (Previously ffmpeg was checked separately as a system
    binary; bundling via imageio-ffmpeg means there's no system dep
    left to track.)
    """
    try:
        import imageio_ffmpeg  # noqa: F401
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _code_is_installed() -> bool:
    """True iff `tree-sitter-language-pack` is importable."""
    try:
        import tree_sitter_language_pack  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_bundled_ffmpeg_on_path() -> str | None:
    """Prepend the imageio-ffmpeg-bundled ffmpeg binary's directory to
    `PATH` so docling / whisper subprocess calls find it.

    Returns the bundled binary's path on success, or `None` when
    `imageio-ffmpeg` isn't installed (the `parsers-asr` extra isn't
    active). Safe to call unconditionally — if a system ffmpeg is
    already on PATH it's left ahead of the bundled one, so bundled
    only wins when there's no system install.

    Called once at process startup (entry point in `__main__` /
    `app.py`) so all downstream subprocess calls inherit the augmented
    PATH.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    binary_dir = str(Path(binary).parent)
    current_path = os.environ.get("PATH", "")
    # Append (not prepend) so a system ffmpeg keeps precedence — bundled
    # only kicks in when nothing else is found. If the bundled dir is
    # already on PATH (e.g. process restarted in same env), don't
    # duplicate the entry.
    parts = current_path.split(os.pathsep)
    if binary_dir not in parts:
        os.environ["PATH"] = current_path + os.pathsep + binary_dir if current_path else binary_dir
    return binary


PARSER_LIFECYCLE_REGISTRY: dict[str, dict[str, Any]] = {
    "asr": {
        # Audio-only path — for a video file we transcribe the audio
        # track but DO NOT capture visual frame content (slides shown
        # on screen are not OCRed). Tracked in
        # [[deferred-video-frame-extraction]].
        # ffmpeg ships bundled via imageio-ffmpeg (declared in the
        # `parsers-asr` extra), so there is NO system-binary dep —
        # installing the extra is the one and only step.
        "display_name": "Audio + video transcription (Whisper Turbo)",
        "pip_extras": "parsers-asr",
        # Libraries that get removed on a full uninstall. We only list
        # ones we own (the extra group itself) - the underlying
        # `docling-slim[format-audio]` deps come along automatically.
        "library_packages": ("openai-whisper", "imageio-ffmpeg"),
        "system_deps": (),
        "system_deps_hints": {},
        "is_installed_fn": _asr_is_installed,
    },
    "code": {
        "display_name": "AST-aware code parsing (tree-sitter)",
        "pip_extras": "parsers-code",
        "library_packages": ("tree-sitter-language-pack",),
        "system_deps": (),
        "system_deps_hints": {},
        "is_installed_fn": _code_is_installed,
    },
}


# ---- system-dep preflight ----


def _check_system_deps(deps: tuple[str, ...]) -> dict[str, str | None]:
    """For each binary name, return its resolved path on PATH (or None)."""
    return {dep: shutil.which(dep) for dep in deps}


def _system_dep_hint(dep: str, hints: dict[str, dict[str, str]]) -> str:
    """Pick the install hint matching the current OS, or fall back to all."""
    per_os = hints.get(dep, {})
    if not per_os:
        return f"install {dep!r} via your system package manager"
    current = platform.system()
    return per_os.get(current, " / ".join(per_os.values()))


# ---- subprocess wrapper (mockable) ----


async def _run_pip(args: list[str], timeout: float = 600.0) -> tuple[bool, str]:
    """Run `python -m pip <args>` async; return (success, combined output).

    `sys.executable` ensures the pip call targets the SAME interpreter
    the app is running under so the new package is visible after a
    restart. Timeout default is generous (10 minutes) because
    `parsers-asr` pulls openai-whisper + numba (large compile chain).

    Uses `asyncio.create_subprocess_exec` so the caller's event loop
    (Flet GUI / CLI / eval harness) stays responsive while pip runs.
    On timeout the child is killed with `proc.kill()` + reaped via
    `proc.wait()` so no zombie remains.
    """
    cmd = [sys.executable, "-m", "pip", *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"pip command timed out after {timeout}s"
        output = (stdout.decode("utf-8", errors="replace") if stdout else "") + (
            stderr.decode("utf-8", errors="replace") if stderr else ""
        )
        return proc.returncode == 0, output
    except Exception as exc:
        return False, f"pip subprocess failed: {exc!r}"


# ---- shared helpers ----


def _registry_entry(name: str) -> dict[str, Any]:
    """Look up a parser-extra in the registry, raise ValueError on miss."""
    entry = PARSER_LIFECYCLE_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown parser extra {name!r}. Known: {sorted(PARSER_LIFECYCLE_REGISTRY)}."
        )
    return entry


# ---- install_parser_extra ----


@dataclass(frozen=True)
class InstallParserExtraPlan:
    """Plan for installing a parser-extra group.

    `system_deps_status` reports whether each required system binary
    is on PATH at plan time. Even when the pip side is already
    installed, missing system deps are surfaced because the parser
    will still fail at ingest time without them.
    """

    extra_name: str
    display_name: str
    pip_extras: str
    already_installed: bool
    system_deps_status: dict[str, str | None]
    system_deps_hints: dict[str, dict[str, str]]

    @property
    def missing_system_deps(self) -> tuple[str, ...]:
        return tuple(d for d, p in self.system_deps_status.items() if p is None)

    @property
    def summary(self) -> str:
        missing = self.missing_system_deps
        sys_hint_block = ""
        if missing:
            hints = "; ".join(
                f"{d}: `{_system_dep_hint(d, self.system_deps_hints)}`" for d in missing
            )
            sys_hint_block = (
                f" System binaries missing on PATH: {', '.join(missing)}. "
                f"Install separately - {hints}."
            )

        if self.already_installed and not missing:
            return f"{self.display_name} is installed and ready."
        if self.already_installed:
            return f"{self.display_name} library is installed.{sys_hint_block}"
        base = (
            f"Install {self.display_name}: runs "
            f"`pip install <this package>[{self.pip_extras}]`. "
            f"Restart required after install."
        )
        return base + sys_hint_block


@dataclass(frozen=True)
class InstallParserExtraResult:
    """Outcome of `install_parser_extra_execute`."""

    extra_name: str
    did_install: bool
    install_ok: bool
    restart_required: bool
    pip_output: str


def install_parser_extra_plan(extra_name: str) -> InstallParserExtraPlan:
    """Build an install plan: check library + system deps."""
    entry = _registry_entry(extra_name)
    return InstallParserExtraPlan(
        extra_name=extra_name,
        display_name=entry["display_name"],
        pip_extras=entry["pip_extras"],
        already_installed=bool(entry["is_installed_fn"]()),
        system_deps_status=_check_system_deps(entry["system_deps"]),
        system_deps_hints=entry["system_deps_hints"],
    )


async def install_parser_extra_execute(
    plan: InstallParserExtraPlan,
    *,
    distribution_name: str = "research-literature-agent",
) -> InstallParserExtraResult:
    """Run `pip install <dist>[<pip_extras>]` if not already installed.

    System binaries (ffmpeg) are NOT installed by this op - they're
    surfaced in the plan summary so the user installs them manually.
    Trying to call out to apt/winget/brew from a Python subprocess
    crosses too many sudo / GUI / permission boundaries to be reliable
    cross-platform.
    """
    if plan.already_installed:
        return InstallParserExtraResult(
            extra_name=plan.extra_name,
            did_install=False,
            install_ok=True,
            restart_required=False,
            pip_output="",
        )

    target = f"{distribution_name}[{plan.pip_extras}]"
    ok, output = await _run_pip(["install", target])
    return InstallParserExtraResult(
        extra_name=plan.extra_name,
        did_install=True,
        install_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- uninstall_parser_extra ----


@dataclass(frozen=True)
class UninstallParserExtraPlan:
    """Plan for uninstalling a parser-extra group's libraries.

    `packages_to_remove` is just the library_packages tuple from the
    registry entry; we don't try to compute transitive removals (pip
    leaves orphan deps behind by design unless you use `pip
    uninstall <dep>` separately).
    """

    extra_name: str
    display_name: str
    packages_to_remove: tuple[str, ...]
    installed: bool

    @property
    def summary(self) -> str:
        if not self.installed:
            return f"{self.display_name} is not installed."
        return (
            f"Uninstall {self.display_name}: runs "
            f"`pip uninstall {' '.join(self.packages_to_remove)}`. "
            f"Restart required after uninstall."
        )


@dataclass(frozen=True)
class UninstallParserExtraResult:
    """Outcome of `uninstall_parser_extra_execute`."""

    extra_name: str
    did_uninstall: bool
    uninstall_ok: bool
    restart_required: bool
    pip_output: str


def uninstall_parser_extra_plan(extra_name: str) -> UninstallParserExtraPlan:
    """Build an uninstall plan."""
    entry = _registry_entry(extra_name)
    return UninstallParserExtraPlan(
        extra_name=extra_name,
        display_name=entry["display_name"],
        packages_to_remove=tuple(entry["library_packages"]),
        installed=bool(entry["is_installed_fn"]()),
    )


async def uninstall_parser_extra_execute(
    plan: UninstallParserExtraPlan,
) -> UninstallParserExtraResult:
    """Pip-uninstall the library packages declared for this parser-extra."""
    if not plan.installed:
        return UninstallParserExtraResult(
            extra_name=plan.extra_name,
            did_uninstall=False,
            uninstall_ok=True,
            restart_required=False,
            pip_output="",
        )
    ok, output = await _run_pip(["uninstall", "-y", *plan.packages_to_remove])
    return UninstallParserExtraResult(
        extra_name=plan.extra_name,
        did_uninstall=True,
        uninstall_ok=ok,
        restart_required=ok,
        pip_output=output,
    )
