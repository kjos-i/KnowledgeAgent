"""Integration tests for `ingestion/parsers/code_parser` — real
tree-sitter AST parsing across multiple languages.

The unit tests in `tests/unit/ingestion/test_code_parser.py` mock
the tree-sitter dependency. This file exercises the real
`tree-sitter-language-pack` grammars against actual source files,
catching API drift (tree-sitter 0.25 method-call API) + grammar
availability.

Requires the `parsers-code` install extra
(`pip install -e ".[parsers-code]"`). Skipped by default; opt in
via `pytest -m integration`. If the extra is not installed, tests
will skip with an informative reason.

Manual interactive counterpart: none — code_parser is exercised by
the pipeline e2e via any ingest of a `.py` / `.js` / etc. file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def _parsers_code_installed() -> bool:
    """Probe for the tree-sitter dep. Skip the file if absent."""
    try:
        import tree_sitter_language_pack  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _parsers_code_installed(),
        reason="parsers-code extra not installed",
    ),
]


PYTHON_SOURCE = '''\
"""Module docstring."""

import os
from typing import Any

CONST = 42


def top_level_function(x: int) -> int:
    """First definition."""
    return x * 2


class Greeter:
    """Second definition."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"


# Trailing module-level code.
if __name__ == "__main__":
    print(top_level_function(5))
'''


JS_SOURCE = """\
// JavaScript example
const PI = 3.14;

function area(r) {
    return PI * r * r;
}

class Circle {
    constructor(r) {
        this.r = r;
    }
    area() {
        return area(this.r);
    }
}

console.log(new Circle(2).area());
"""


def test_code_parser_python_yields_definition_chunks(tmp_path: Path) -> None:
    """A real Python file with 2 top-level definitions (function +
    class) yields at least 2 code chunks plus any module-level
    glue chunks."""
    from knowledge_agent.ingestion.parsers import code_parser

    src = tmp_path / "sample.py"
    src.write_text(PYTHON_SOURCE)

    chunks = code_parser.parse(src)
    assert len(chunks) >= 2
    # Every chunk has content_type = "code".
    assert all(c.content_type == "code" for c in chunks)
    # All chunks have non-empty text.
    assert all(c.text for c in chunks)
    # The function + class definitions appear in some chunk's text.
    text_all = "\n".join(c.text for c in chunks)
    assert "top_level_function" in text_all
    assert "class Greeter" in text_all


def test_code_parser_javascript_yields_definition_chunks(tmp_path: Path) -> None:
    """A real JavaScript file also parses via tree-sitter-language-
    pack's grammar — catches grammar availability + node-type
    declarations in the parser."""
    from knowledge_agent.ingestion.parsers import code_parser

    src = tmp_path / "sample.js"
    src.write_text(JS_SOURCE)

    chunks = code_parser.parse(src)
    assert len(chunks) >= 1
    assert all(c.content_type == "code" for c in chunks)
    text_all = "\n".join(c.text for c in chunks)
    assert "function area" in text_all
    assert "class Circle" in text_all


def test_code_parser_single_line_file_yields_one_chunk(tmp_path: Path) -> None:
    """A trivial file (one expression, no definitions) still yields
    one chunk containing the whole file — the documented fallback."""
    from knowledge_agent.ingestion.parsers import code_parser

    src = tmp_path / "trivial.py"
    src.write_text("x = 1\n")

    chunks = code_parser.parse(src)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "x = 1"


def test_code_parser_chunk_indices_are_sequential(tmp_path: Path) -> None:
    """chunk_index is 0-based + sequential across the emitted
    chunks — required for downstream make_chunk_id()."""
    from knowledge_agent.ingestion.parsers import code_parser

    src = tmp_path / "sample.py"
    src.write_text(PYTHON_SOURCE)
    chunks = code_parser.parse(src)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
