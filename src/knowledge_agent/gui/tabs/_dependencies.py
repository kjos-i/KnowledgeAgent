"""Generated 'Dependencies' body for the global Info tab.

Lists the packages Knowledge Agent declares, read live from the installed package
metadata (`importlib.metadata`) — the SAME data pip installed from `pyproject.toml`.
So the list can't drift from the real dependencies and needs no hand-maintained
copy (single source of truth). Grouped into core + each extra; the version shown is
what's installed in this environment.
"""

from __future__ import annotations

import importlib.metadata as im
import re

_DIST = "knowledge-agent"
_EXTRA_RE = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _req_name(req: str) -> str | None:
    m = _NAME_RE.match(req)
    return m.group(1) if m else None


def _req_extra(req: str) -> str | None:
    """The extra a requirement belongs to (from its `; extra == "x"` marker), or
    None for a core dependency."""
    if ";" not in req:
        return None
    m = _EXTRA_RE.search(req.split(";", 1)[1])
    return m.group(1) if m else None


def _version(name: str) -> str:
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return "not installed"


def dependencies_markdown() -> str:
    """Markdown listing the declared dependencies + their installed versions,
    grouped core → extras. Falls back to a hint when the package metadata isn't
    available (running from an un-installed source tree)."""
    try:
        reqs = im.requires(_DIST) or []
        app_version = im.version(_DIST)
    except im.PackageNotFoundError:
        return (
            "# Dependencies\n\nPackage metadata isn't available — install the app "
            "(`pip install -e .`) to list its dependencies here."
        )

    core: list[str] = []
    extras: dict[str, list[str]] = {}
    for req in reqs:
        name = _req_name(req)
        if not name:
            continue
        line = f"- **{name}** — {_version(name)}"
        extra = _req_extra(req)
        if extra is None:
            core.append(line)
        else:
            extras.setdefault(extra, []).append(line)

    out: list[str] = [
        "# Dependencies",
        "",
        f"Knowledge Agent {app_version} is built on these open-source packages. "
        "The list is read live from the installed package metadata, so it always "
        "matches the real dependencies; versions shown are what's installed here.",
        "",
        "## Core",
        "",
        *(core or ["- (none declared)"]),
    ]
    for extra in sorted(extras):
        out += ["", f"## Extra: {extra}", "", *extras[extra]]
    return "\n".join(out)
