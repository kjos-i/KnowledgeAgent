"""Tests for the AST-aware source-code parser strategy.

Mocks the lazy tree-sitter import (`_get_parser`) to return a fake
parser yielding a hand-rolled AST. This way the tests cover the chunk-
emission logic without needing `tree-sitter-language-pack` installed.

Coverage:
  - EXTENSIONS + _EXTENSION_TO_LANGUAGE coherent
  - dispatcher routes .py/.ts/etc. into code_parser
  - definitions emitted as one chunk each
  - inter-definition code (imports, top-level statements) becomes its
    own chunk
  - file with no recognised definitions emits the whole file as one
    chunk
  - oversized chunks split greedy-line-aware
  - content_type = "code" everywhere; chunk_index monotonic
  - missing `parsers-code` extra raises a clear ImportError
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from knowledge_agent.ingestion.parsers import code_parser

# ---- fake AST node + parser ----
#
# Duck-types the tree-sitter 0.25 Node API where every accessor is a
# METHOD call: `node.kind()` (renamed from `.type`), `node.start_byte()`,
# `node.end_byte()`, `node.child_count()`, `node.child(i)`. Same shape
# for the Tree (`tree.root_node()`).


class _FakeNode:
    def __init__(
        self,
        kind_value: str,
        start_byte_value: int,
        end_byte_value: int,
        children: list["_FakeNode"] | None = None,
    ) -> None:
        self._kind = kind_value
        self._start = start_byte_value
        self._end = end_byte_value
        self._children = children or []

    def kind(self) -> str:
        return self._kind

    def start_byte(self) -> int:
        return self._start

    def end_byte(self) -> int:
        return self._end

    def child_count(self) -> int:
        return len(self._children)

    def child(self, index: int) -> "_FakeNode":
        return self._children[index]


class _FakeTree:
    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def root_node(self) -> _FakeNode:
        return self._root


class _FakeParser:
    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def parse(self, _source_str: str) -> _FakeTree:
        return _FakeTree(root=self._root)


def _patch_parser(root: _FakeNode):
    """Patch `_get_parser` to return a fake parser yielding `root`."""
    # Clear lru_cache so each test gets a fresh parser
    code_parser._get_parser.cache_clear()
    return patch.object(code_parser, "_get_parser", return_value=_FakeParser(root))


# ---- EXTENSIONS + language map ----


def test_extensions_covers_eight_core_languages():
    """Sanity check: launch set of extensions is present."""
    exts = set(code_parser.EXTENSIONS)
    expected = {"py", "ts", "tsx", "js", "go", "rs", "java", "c", "cpp"}
    assert expected.issubset(exts), f"missing: {expected - exts}"


def test_extensions_sorted():
    """EXTENSIONS is sorted for deterministic test output."""
    assert list(code_parser.EXTENSIONS) == sorted(code_parser.EXTENSIONS)


def test_definition_node_types_covers_all_supported_languages():
    """Every language in _EXTENSION_TO_LANGUAGE has definition node types.

    Catches accidental drift between adding an extension and registering
    its definition types.
    """
    languages_used = set(code_parser._EXTENSION_TO_LANGUAGE.values())
    declared = set(code_parser._DEFINITION_NODE_TYPES.keys())
    assert languages_used.issubset(declared), (
        f"languages missing _DEFINITION_NODE_TYPES: {languages_used - declared}"
    )


# ---- AST chunking ----


def test_two_python_functions_emit_one_chunk_each(tmp_path: Path):
    foo_src = b"def foo():\n    return 1\n"
    sep = b"\n"
    bar_src = b"def bar():\n    return 2\n"
    source = foo_src + sep + bar_src
    path = tmp_path / "x.py"
    path.write_bytes(source)

    # Offsets derived from len() so the test doesn't depend on manual
    # arithmetic. The single \n between functions is whitespace and
    # gets skipped by the prelude-strip; we expect exactly 2 chunks.
    foo_end = len(foo_src)
    bar_start = foo_end + len(sep)
    root = _FakeNode(
        "module",
        0,
        len(source),
        children=[
            _FakeNode("function_definition", 0, foo_end),
            _FakeNode("function_definition", bar_start, len(source)),
        ],
    )
    with _patch_parser(root):
        chunks = code_parser.parse(path)

    assert len(chunks) == 2
    assert "def foo" in chunks[0].text
    assert "def bar" in chunks[1].text
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert all(c.content_type == "code" for c in chunks)


def test_inter_definition_code_emits_its_own_chunk(tmp_path: Path):
    """Imports + module-level code before/between definitions get a chunk."""
    source = b"import os\n\ndef foo():\n    return os.getcwd()\n"
    path = tmp_path / "x.py"
    path.write_bytes(source)

    # imports occupy bytes 0..10 (b"import os\n"); foo starts at 11
    root = _FakeNode(
        "module",
        0,
        len(source),
        children=[
            _FakeNode("function_definition", 11, len(source)),
        ],
    )
    with _patch_parser(root):
        chunks = code_parser.parse(path)

    assert len(chunks) == 2
    assert "import os" in chunks[0].text
    assert "def foo" in chunks[1].text


def test_file_with_no_definitions_emits_whole_file(tmp_path: Path):
    """A script with only top-level statements becomes one chunk."""
    source = b"print('hello')\nx = 1 + 2\nprint(x)\n"
    path = tmp_path / "script.py"
    path.write_bytes(source)

    root = _FakeNode(
        "module",
        0,
        len(source),
        children=[],
    )
    with _patch_parser(root):
        chunks = code_parser.parse(path)

    assert len(chunks) == 1
    assert chunks[0].text == source.decode("utf-8")
    assert chunks[0].content_type == "code"


# ---- oversized splitting ----


def test_split_oversized_leaves_small_text_alone():
    text = "def foo():\n    return 1\n"
    assert code_parser._split_oversized(text) == [text]


def test_split_oversized_breaks_long_text_on_line_boundaries():
    # Build a function whose source exceeds _MAX_CHARS by ~3x
    lines = [f"    x = {i}" for i in range(800)]
    text = "def big():\n" + "\n".join(lines)
    pieces = code_parser._split_oversized(text)

    assert len(pieces) >= 2, "expected at least two pieces"
    # Every piece stays at or under the cap
    for p in pieces:
        assert len(p) <= code_parser._MAX_CHARS, f"piece exceeded cap: {len(p)}"
    # Round-trip: rejoining should reconstruct the original (modulo
    # the single newline we inserted between pieces during split)
    rejoined = "\n".join(pieces)
    assert rejoined == text


# ---- dispatcher integration ----


def _dispatcher_config():
    """CorpusConfig instance whose fields the code parser ignores.

    parse_document takes a CorpusConfig even though non-docling
    parsers don't read from it; we pass a minimal instance so the
    signature is satisfied.
    """
    from knowledge_agent.corpus_config import CorpusConfig, LayerFlags

    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),
    )


def test_dispatcher_routes_py_to_code_parser(tmp_path: Path):
    """End-to-end: the top-level parse_document picks code_parser."""
    from knowledge_agent.ingestion.parse import parse_document

    source = b"def hi():\n    pass\n"
    path = tmp_path / "x.py"
    path.write_bytes(source)
    root = _FakeNode(
        "module",
        0,
        len(source),
        children=[_FakeNode("function_definition", 0, len(source))],
    )
    with _patch_parser(root):
        chunks = parse_document(path, _dispatcher_config())

    assert len(chunks) == 1
    assert chunks[0].content_type == "code"


def test_dispatcher_routes_ts_to_code_parser(tmp_path: Path):
    from knowledge_agent.ingestion.parse import parse_document

    source = b"function hi() { return 1; }\n"
    path = tmp_path / "x.ts"
    path.write_bytes(source)
    root = _FakeNode(
        "program",
        0,
        len(source),
        children=[_FakeNode("function_declaration", 0, len(source))],
    )
    with _patch_parser(root):
        chunks = parse_document(path, _dispatcher_config())

    assert len(chunks) == 1


def test_supported_extensions_now_includes_code_family():
    from knowledge_agent.ingestion.parse import supported_extensions

    exts = supported_extensions()
    for code_ext in ("py", "ts", "tsx", "js", "go", "rs", "java", "c", "cpp"):
        assert code_ext in exts, f"missing {code_ext!r}"


# ---- missing optional extra ----


def test_parse_raises_import_error_when_extra_missing(tmp_path: Path):
    """Without `parsers-code`, parse() should fail clearly pointing at the extra.

    Simulated by clearing the lru_cache and making the lazy import raise.
    """
    path = tmp_path / "x.py"
    path.write_bytes(b"def foo(): pass\n")

    code_parser._get_parser.cache_clear()
    with (
        patch.dict("sys.modules", {"tree_sitter_language_pack": None}),
        pytest.raises(ImportError, match="parsers-code"),
    ):
        code_parser.parse(path)
