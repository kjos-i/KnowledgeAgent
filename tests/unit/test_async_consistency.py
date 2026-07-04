"""Static check: no sync function calls an async API without `await`.

Regression guard for the class of bug surfaced during the Phase 1
async refactor (2026-06-30). A sync `def` function calling an
`async def` callable just constructs a coroutine and returns it —
no exception. The failure surfaces later (e.g. `coroutine[0]`
raises `TypeError: 'coroutine' object is not subscriptable`) at
the boundary catch, far from the actual mistake site. The check
fails fast at lint time instead.

The authoritative list of async names is auto-derived: we walk the
modules in `_ASYNC_BEARING_MODULES`, collect every top-level
`async def` and every async method on any class, and that's the
set every sync call site must avoid. Adding a new async API to
one of those modules is auto-picked-up — no test edit needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "knowledge_agent"


_ASYNC_BEARING_MODULES = [
    _SRC_ROOT / "_http_client.py",
    _SRC_ROOT / "cli.py",
    _SRC_ROOT / "health.py",
    _SRC_ROOT / "gui" / "app.py",
    _SRC_ROOT / "kg" / "client.py",
    _SRC_ROOT / "search" / "client.py",
    _SRC_ROOT / "ingestion" / "metadata.py",
    _SRC_ROOT / "ingestion" / "pipeline.py",
    _SRC_ROOT / "ingestion" / "bulk_ops.py",
    _SRC_ROOT / "ingestion" / "metadata_resolution.py",
    _SRC_ROOT / "ingestion" / "parser_lifecycle.py",
    _SRC_ROOT / "kg" / "ontology_helpers.py",
    _SRC_ROOT / "kg" / "ontology_fibo_writes.py",
    _SRC_ROOT / "kg" / "ontology_lifecycle.py",
    _SRC_ROOT / "entity_extractors" / "extractor_lifecycle.py",
    _SRC_ROOT / "embedder_lifecycle.py",
    _SRC_ROOT / "llm_lifecycle.py",
    _SRC_ROOT / "embedder_factory.py",
]

# Names to exempt even if they show up as `async def` in one module —
# typically because the SAME name is also a sync helper elsewhere
# (legitimate name collision; the sync caller is calling the sync
# helper, not the async one).
_IGNORE_NAMES: set[str] = set()
# Add to this set if a false positive appears — e.g. a sync helper
# elsewhere has the same name as an async method here, and the test
# misidentifies a legitimate sync-call-to-sync-helper as a bug.


_SCAN_ROOTS = [_SRC_ROOT, _REPO_ROOT / "scripts"]


def _collect_async_names() -> set[str]:
    """Walk `_ASYNC_BEARING_MODULES` and return every async fn/method name."""
    names: set[str] = set()
    for path in _ASYNC_BEARING_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                names.add(node.name)
    return names - _IGNORE_NAMES


def _is_asyncio_run_call(node: ast.Call) -> bool:
    """True iff this Call is `asyncio.run(...)` — the canonical sync→async
    bridge. Used to exempt the inner call from the sync-calls-async check.
    """
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "run"
        and isinstance(func.value, ast.Name)
        and func.value.id == "asyncio"
    )


class _SyncCallsFinder(ast.NodeVisitor):
    """Walk a module; record any sync-scope call whose name is async."""

    def __init__(self, async_names: set[str]) -> None:
        self._async_names = async_names
        # Stack of bools. True = currently inside a `def` (sync). The
        # module body is NOT pushed onto the stack — module-level
        # statements aren't a place we await anything anyway, and many
        # legitimate patterns (decorators, dispatcher tables) reference
        # async-named callables at import time.
        self._sync_stack: list[bool] = []
        # Depth counter: > 0 means we're processing args of an
        # `asyncio.run(...)` call, where an "unawaited" coroutine is the
        # whole point (`asyncio.run(coro())` is the sync→async bridge).
        self._asyncio_run_depth = 0
        self.problems: list[tuple[int, str]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._sync_stack.append(False)
        self.generic_visit(node)
        self._sync_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._sync_stack.append(True)
        self.generic_visit(node)
        self._sync_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambdas are sync; treat like a sync def for nesting.
        self._sync_stack.append(True)
        self.generic_visit(node)
        self._sync_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self._sync_stack and self._sync_stack[-1] and self._asyncio_run_depth == 0:
            name: str | None = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name is not None and name in self._async_names:
                self.problems.append((node.lineno, name))
        # `asyncio.run(<coro_call>)`: descend with the depth flag set so
        # the inner coroutine-producing call isn't flagged.
        if _is_asyncio_run_call(node):
            self._asyncio_run_depth += 1
            try:
                self.generic_visit(node)
            finally:
                self._asyncio_run_depth -= 1
        else:
            self.generic_visit(node)


def test_no_sync_function_calls_async_api_without_await() -> None:
    """Every call to a known-async function/method must sit inside an
    `async def` so it can be awaited.

    When this test fails, the message lists each offender by
    `path:line: sync call to async <name>`. Fix by either:
      - converting the enclosing `def` to `async def` and adding
        `await`, OR
      - wrapping with `asyncio.to_thread(...)` (sync wrapper around
        async helper) or `asyncio.run(...)` (sync entry point).

    To add a new async API: just write `async def X` in one of the
    `_ASYNC_BEARING_MODULES`. The test auto-picks it up.
    """
    async_names = _collect_async_names()
    offenders: list[str] = []
    for root in _SCAN_ROOTS:
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in str(py):
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"))
            finder = _SyncCallsFinder(async_names)
            finder.visit(tree)
            for lineno, name in finder.problems:
                rel = py.relative_to(_REPO_ROOT)
                offenders.append(f"  {rel.as_posix()}:{lineno}: sync call to async {name!r}")

    if offenders:
        pytest.fail(
            "Sync function calls an async API without `await`. These "
            "produce coroutines instead of the expected return values "
            "and fail later with cryptic 'coroutine' errors at the "
            "boundary catch:\n\n" + "\n".join(offenders) + "\n\nFix: see this test's docstring."
        )
