"""Tests for Create New Dataset — corpus path derivation + validation.

The folder field pairs with a mode radio (`_corpus_folder`):

  * create (default) → the corpus lives in a `<location>/<name>/`
    subfolder whose `lancedb/`, `corpus.toml`, `figures/` sit side by side.
  * adopt            → the picked folder *is* the corpus home; artefacts
    land directly in it (adopts an existing folder in place).

These tests pin both derivations (no double-nesting, no auto-fill), the
pre-create guards (which also make adopt-mode safe), and the cosmetic
relabel on mode change. The IO boundaries (Neo4j ping, keyring, disk
config, env bridge) are mocked; only the path logic + guards run for real.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.config_store import CorpusEntry, GuiConfig
from knowledge_agent.gui.library.create_new_dataset import CreateNewDatasetTab

if TYPE_CHECKING:
    from pathlib import Path

_MOD = "knowledge_agent.gui.library.create_new_dataset"


def _tab() -> CreateNewDatasetTab:
    app = MagicMock()
    app.gui_config = GuiConfig()  # real list for `.corpora`, real Path fields
    app.page = MagicMock()
    return CreateNewDatasetTab(app)


def _fill(
    tab: CreateNewDatasetTab,
    *,
    name: str = "c1",
    uri: str = "neo4j://127.0.0.1:7687",
    user: str = "neo4j",
    password: str = "pw",  # dummy throwaway for the form — never a real secret
    folder: str = "",
) -> None:
    tab.name_field.value = name
    tab.uri_field.value = uri
    tab.user_field.value = user
    tab.password_field.value = password
    tab.folder_field.value = folder


# ---- _validate: required fields ----------------------------------------


def test_validate_requires_name():
    tab = _tab()
    _fill(tab, name="", folder="/somewhere")
    ok, msg = tab._validate()
    assert not ok and "name is required" in msg


def test_validate_requires_folder():
    tab = _tab()
    _fill(tab, name="c1", folder="")
    ok, msg = tab._validate()
    assert not ok and "folder is required" in msg


def test_validate_rejects_duplicate_name(tmp_path: Path):
    tab = _tab()
    tab.app.gui_config.corpora = [
        CorpusEntry(
            name="dup",
            neo4j_uri="neo4j://h:7687",
            lancedb_path=tmp_path / "l",
            corpus_config_path=tmp_path / "c.toml",
        )
    ]
    _fill(tab, name="dup", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert not ok and "already registered" in msg


# ---- _validate: path-derivation guards ---------------------------------


def test_validate_rejects_existing_corpus_toml(tmp_path: Path):
    """A corpus.toml already at `<location>/<name>/` would be clobbered."""
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "corpus.toml").write_text("x", encoding="utf-8")
    tab = _tab()
    _fill(tab, name="c1", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert not ok and "already exists" in msg


def test_validate_rejects_nonempty_lancedb(tmp_path: Path):
    ldb = tmp_path / "c1" / "lancedb"
    ldb.mkdir(parents=True)
    (ldb / "data.lance").write_text("x", encoding="utf-8")
    tab = _tab()
    _fill(tab, name="c1", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert not ok and "non-empty" in msg


def test_validate_ok_for_fresh_location(tmp_path: Path):
    tab = _tab()
    _fill(tab, name="fresh", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert ok and msg == ""


# ---- on_create_clicked: path derivation + registration -----------------


def test_on_create_derives_subfolder_paths_and_registers(tmp_path: Path, monkeypatch):
    """The corpus lands in `<location>/<name>/` with lancedb + corpus.toml
    side by side — no double-nesting. The registered CorpusEntry + the
    top-level GuiConfig mirror carry those derived paths."""
    tab = _tab()
    tab.app.chat_panel = MagicMock()
    tab.app.library_tab = MagicMock()
    _fill(tab, name="mycorpus", folder=str(tmp_path))
    # Isolate the handler's direct `os.environ["NEO4J_PASSWORD"] = ...`
    # write so it doesn't leak into other tests (monkeypatch restores on
    # teardown regardless of the later in-handler mutation).
    monkeypatch.setenv("NEO4J_PASSWORD", "")

    with (
        patch.object(tab, "_ping_neo4j", AsyncMock(return_value=None)),
        patch(f"{_MOD}._write_corpus_toml") as write_toml,
        patch(f"{_MOD}.set_corpus_password"),
        patch(f"{_MOD}.save_config"),
        patch(f"{_MOD}.apply_connection_to_env"),
        patch(f"{_MOD}.reset_after_key_change"),
    ):
        asyncio.run(tab.on_create_clicked(MagicMock()))

    expected = tmp_path / "mycorpus"
    # corpus.toml is written at <location>/<name>/corpus.toml.
    write_toml.assert_called_once()
    assert write_toml.call_args.args[0] == expected / "corpus.toml"
    # The registry entry + active mirror carry the derived subfolder paths.
    entry = tab.app.gui_config.corpora[-1]
    assert entry.name == "mycorpus"
    assert entry.lancedb_path == expected / "lancedb"
    assert entry.corpus_config_path == expected / "corpus.toml"
    assert tab.app.gui_config.active_corpus_name == "mycorpus"
    assert tab.app.gui_config.lancedb_path == expected / "lancedb"
    assert tab.app.gui_config.corpus_config_path == expected / "corpus.toml"


def test_on_create_validation_failure_short_circuits(tmp_path: Path):
    """A bad form fails fast: the async Neo4j ping is never reached."""
    tab = _tab()
    _fill(tab, name="", folder=str(tmp_path))
    with patch.object(tab, "_ping_neo4j", AsyncMock()) as ping:
        asyncio.run(tab.on_create_clicked(MagicMock()))
    ping.assert_not_called()
    assert tab.status.value.startswith("cannot create")


# ---- folder mode: create (subfolder) vs adopt (folder itself) ----------


def test_create_mode_derives_named_subfolder(tmp_path: Path):
    """Default mode makes a `<location>/<name>` subfolder."""
    tab = _tab()  # radio defaults to create
    assert tab._folder_mode() == "create"
    assert tab._corpus_folder(str(tmp_path), "c1") == tmp_path / "c1"


def test_adopt_mode_uses_selected_folder_directly(tmp_path: Path):
    """Adopt mode treats the picked folder as the corpus home — no subfolder."""
    tab = _tab()
    tab.location_mode_radio.value = "adopt"
    assert tab._folder_mode() == "adopt"
    assert tab._corpus_folder(str(tmp_path), "c1") == tmp_path
    # A fresh empty folder is valid to adopt.
    _fill(tab, name="c1", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert ok and msg == ""


def test_adopt_mode_rejects_folder_that_is_already_a_corpus(tmp_path: Path):
    """The corpus.toml guard makes adopt safe: refuse an existing corpus."""
    (tmp_path / "corpus.toml").write_text("x", encoding="utf-8")
    tab = _tab()
    tab.location_mode_radio.value = "adopt"
    _fill(tab, name="c1", folder=str(tmp_path))
    ok, msg = tab._validate()
    assert not ok and "already exists" in msg


def test_adopt_mode_registers_folder_as_corpus_home(tmp_path: Path, monkeypatch):
    """on_create in adopt mode writes corpus.toml + registers the picked
    folder itself — lancedb/corpus.toml sit directly inside it, not in a
    `<name>` subfolder."""
    tab = _tab()
    tab.location_mode_radio.value = "adopt"
    tab.app.chat_panel = MagicMock()
    tab.app.library_tab = MagicMock()
    _fill(tab, name="mycorpus", folder=str(tmp_path))
    monkeypatch.setenv("NEO4J_PASSWORD", "")

    with (
        patch.object(tab, "_ping_neo4j", AsyncMock(return_value=None)),
        patch(f"{_MOD}._write_corpus_toml") as write_toml,
        patch(f"{_MOD}.set_corpus_password"),
        patch(f"{_MOD}.save_config"),
        patch(f"{_MOD}.apply_connection_to_env"),
        patch(f"{_MOD}.reset_after_key_change"),
    ):
        asyncio.run(tab.on_create_clicked(MagicMock()))

    # corpus.toml written directly in the picked folder — not tmp/mycorpus.
    write_toml.assert_called_once()
    assert write_toml.call_args.args[0] == tmp_path / "corpus.toml"
    entry = tab.app.gui_config.corpora[-1]
    assert entry.lancedb_path == tmp_path / "lancedb"
    assert entry.corpus_config_path == tmp_path / "corpus.toml"


def test_mode_change_relabels_folder_field_and_structure_tree():
    """Toggling the radio swaps the folder caption/hint + the right-pane tree.

    The folder caption is the labeled_field row's caption Text (with a
    trailing colon), not a floating TextField `label` anymore.
    """
    tab = _tab()
    # Flip to adopt.
    tab.location_mode_radio.value = "adopt"
    tab._on_location_mode_change(MagicMock())
    assert tab._folder_caption.value == "Corpus folder:"
    assert "BE this corpus" in tab.folder_field.hint_text
    assert "<selected folder>" in tab._structure_tree.value
    # Flip back to create.
    tab.location_mode_radio.value = "create"
    tab._on_location_mode_change(MagicMock())
    assert tab._folder_caption.value == "Location:"
    assert "<corpus name>" in tab._structure_tree.value
