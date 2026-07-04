"""Source-code parser strategy (AST-aware chunking via tree-sitter).

Walks the AST of a source-code file, emits one chunk per top-level
function / class / type definition, plus separate chunks for any code
that lives between or around those definitions (imports, module-level
statements, trailing code). Definitions larger than `_MAX_CHARS` are
naive-line-split as a follow-up pass; AST-aware sub-function splitting
is a future refinement.

`content_type` is "code" on every emitted chunk.

Heavy deps (`tree-sitter`, `tree-sitter-language-pack`) are imported
lazily inside `parse()` so this module loads cleanly even when the
`parsers-code` install extra hasn't been installed. The dispatcher's
`supported_extensions()` only reads `EXTENSIONS` (a tuple of strings) -
no heavy deps touched. First `.py` file ingested = first `tree_sitter`
import; clear `ImportError` pointing at the `parsers-code` extra if it's
missing. Preflight wiring lives in `parser_lifecycle.py`.

Adding a new language = one entry in `_EXTENSION_TO_LANGUAGE` + one
entry in `_DEFINITION_NODE_TYPES`. tree-sitter-language-pack ships
grammars for ~305 languages, so coverage is wide; we just need to
declare which AST node types count as "definition boundaries" per
language we support.

Files in a supported extension but with no recognizable definitions
(e.g., a simple script with only top-level statements) yield a single
chunk containing the whole file, then naive-line-split if oversized.

tree-sitter 0.25 API note: every Node accessor is a METHOD call, not
an attribute - `node.kind()` (was `.type`), `node.start_byte()`,
`node.end_byte()`, `node.child_count()`, `node.child(i)`. Parser input
is a `str`, NOT `bytes`. Byte offsets returned are UTF-8 byte positions,
so we encode + slice the source as bytes when extracting chunk text.
"""

import logging
from functools import cache
from pathlib import Path
from typing import Any

from knowledge_agent.ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

# Maximum chars per chunk before naive line-splitting kicks in.
# 1500 is the LlamaIndex CodeSplitter default; it lines up roughly with
# 400-500 tokens for typical code, well under embedding-model limits.
_MAX_CHARS = 1500

# Extension -> tree-sitter language name. Adding a language here is half
# the work; the other half is _DEFINITION_NODE_TYPES below.
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    "py": "python",
    "ts": "typescript",
    "tsx": "tsx",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "c": "c",
    "h": "c",  # ambiguous: could be C or C++ header; default to C
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "hxx": "cpp",
}

# Per-language AST node types that count as definition boundaries.
# Node-type names come from each grammar's `grammar.js`; verify with
# tree-sitter's playground (https://tree-sitter.github.io/tree-sitter/playground)
# before adding new entries.
_DEFINITION_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "function_definition",
            "class_definition",
            "decorated_definition",
        }
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        }
    ),
    "javascript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "method_definition",
        }
    ),
    "go": frozenset(
        {
            "function_declaration",
            "method_declaration",
            "type_declaration",
        }
    ),
    "rust": frozenset(
        {
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
        }
    ),
    "java": frozenset(
        {
            "class_declaration",
            "method_declaration",
            "interface_declaration",
            "enum_declaration",
        }
    ),
    "c": frozenset(
        {
            "function_definition",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        }
    ),
    "cpp": frozenset(
        {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "namespace_definition",
        }
    ),
}

# Parser dispatcher contract.
EXTENSIONS: tuple[str, ...] = tuple(sorted(_EXTENSION_TO_LANGUAGE.keys()))


def parse(path: Path) -> list[ParsedChunk]:
    """Parse one source-code file into AST-aware chunks.

    Raises a clear `ImportError` if the `parsers-code` extra isn't
    installed. Any tree-sitter parse errors bubble up to the caller.
    """
    ext = path.suffix.lower().lstrip(".")
    language = _EXTENSION_TO_LANGUAGE.get(ext)
    if language is None:
        raise ValueError(f"code_parser does not handle extension {ext!r}")

    parser = _get_parser(language)
    # tree-sitter 0.25 wants str input. Read as UTF-8; byte offsets
    # the AST reports refer to UTF-8 byte positions, so we encode for
    # slicing the chunk text out.
    source_str = path.read_text(encoding="utf-8")
    source_bytes = source_str.encode("utf-8")
    tree = parser.parse(source_str)

    definition_types = _DEFINITION_NODE_TYPES.get(language, frozenset())
    raw_chunks = _split_by_definitions(tree.root_node(), source_bytes, definition_types)

    chunks: list[ParsedChunk] = []
    for text in raw_chunks:
        for piece in _split_oversized(text):
            chunks.append(
                ParsedChunk(
                    chunk_index=len(chunks),
                    text=piece,
                    content_type="code",
                )
            )
    logger.info("parsed %s (%s) -> %d chunks", path.name, language, len(chunks))
    return chunks


@cache
def _get_parser(language: str) -> Any:
    """Load and cache a tree-sitter parser for one language.

    Lazy import so the module loads even without the `parsers-code`
    install extra. Wrapped ImportError points the user at the extra
    by name.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        raise ImportError(
            "code_parser requires the 'parsers-code' install extra. "
            "Install with: pip install research-literature-agent[parsers-code]"
        ) from exc
    return get_parser(language)


def _split_by_definitions(
    root: Any,
    source_bytes: bytes,
    definition_types: frozenset[str],
) -> list[str]:
    """Walk top-level children; emit definitions + group inter-def code.

    For a file with no recognised definitions (or an empty
    `definition_types` set), returns one chunk containing the whole
    file - the caller's oversized-split pass handles it from there.

    All Node accessors are method calls (tree-sitter 0.25 API):
    `.kind()`, `.start_byte()`, `.end_byte()`, `.child_count()`,
    `.child(i)`.
    """
    if not definition_types:
        return [source_bytes.decode("utf-8", errors="replace")]

    chunks: list[str] = []
    cursor = root.start_byte()
    for i in range(root.child_count()):
        child = root.child(i)
        if child.kind() not in definition_types:
            continue
        child_start = child.start_byte()
        child_end = child.end_byte()
        # Flush any code between the previous boundary and this definition
        # (imports, module-level statements, etc.) as its own chunk.
        if cursor < child_start:
            prelude = source_bytes[cursor:child_start].decode("utf-8", errors="replace")
            if prelude.strip():
                chunks.append(prelude)
        chunks.append(source_bytes[child_start:child_end].decode("utf-8", errors="replace"))
        cursor = child_end
    # Trailing code after the last definition.
    root_end = root.end_byte()
    if cursor < root_end:
        trailing = source_bytes[cursor:root_end].decode("utf-8", errors="replace")
        if trailing.strip():
            chunks.append(trailing)
    # File with no definitions -> whole file as one chunk.
    if not chunks:
        chunks = [source_bytes.decode("utf-8", errors="replace")]
    return chunks


def _split_oversized(text: str) -> list[str]:
    """Naive line-aware split when a single chunk exceeds `_MAX_CHARS`.

    Greedy packing: accumulate lines until the next one would push us
    over the limit, then start a new chunk. Always emits at least one
    chunk (even for empty input - the caller filters empties earlier).
    """
    if len(text) <= _MAX_CHARS:
        return [text]
    lines = text.split("\n")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        # +1 for the newline we'll re-add when joining
        projected = current_len + len(line) + (1 if current else 0)
        if projected > _MAX_CHARS and current:
            pieces.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len = projected
    if current:
        pieces.append("\n".join(current))
    return pieces
