"""Library → Create New Dataset sub-tab — register a new corpus.

Storage-only form:

  Corpus name, Neo4j URI / user / password, a folder field, and a
  radio that chooses how that folder is used:

  * "Create a new folder named after the corpus" (default) — the
    folder field is a *parent* location; a `<name>` subfolder is made
    inside it and holds all the corpus's artefacts.
  * "Use the selected folder as the corpus home" — the folder field
    *is* the corpus home; artefacts land directly in it (adopts a
    folder you've already made, e.g. one that already holds a
    `documents/` subfolder).

The GUI derives the corpus home from the folder field + the radio
(see `_corpus_folder`), then the internal paths from that home:

  * Corpus folder     = `<folder>/<name>` (create) | `<folder>` (adopt)
  * LanceDB path      = `<corpus folder>/lancedb`
  * corpus.toml path  = `<corpus folder>/corpus.toml`
  * figures dir       = `<corpus folder>/figures`

Users don't need to think about file structure — they pick a folder
plus a mode, and we manage what goes where inside the corpus home.

On [Create corpus]:

  1. Validate the form (see `_validate` below).
  2. Ping the Neo4j URI with a 5 s timeout.
  3. Write a fresh corpus.toml at `<corpus folder>/corpus.toml`
     with seed defaults.
  4. Save the Neo4j password to the OS keyring under
     `f"neo4j-{name}"`.
  5. Append a `CorpusEntry` to `GuiConfig.corpora`; set
     `active_corpus_name` to the new name; mirror URI / user / paths
     to the top-level `GuiConfig` fields; bridge to env; drop
     cached factories.
  6. Chat-panel status message + a hint to open Select Dataset to
     tune the corpus's config.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft
import tomlkit

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    CorpusEntry,
    apply_connection_to_env,
    save_config,
    set_corpus_password,
)
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.kg.corpus_config import CorpusConfig

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


# Neo4j ping timeout for the validation step. 5 s is long enough for
# a slow local DBMS boot but short enough that a wrong URI fails
# quickly.
_NEO4J_PING_TIMEOUT = 5.0


# Split proportions for the internal 2-column layout — form on the left,
# helper info on the right. Fixed (no dragger) — the right column exists
# only to organise the content and give future info a home.
_LEFT_FLEX = 60
_RIGHT_FLEX = 40


# Folder-mode radio values + the copy that swaps when the mode flips.
# 'create' (default) treats the folder field as a parent and makes a
# `<name>` subfolder inside it; 'adopt' treats the folder field as the
# corpus home itself. The actual path branch lives in `_corpus_folder`;
# everything below is cosmetic (field label/hint + the right-pane tree).
_MODE_CREATE = "create"
_MODE_ADOPT = "adopt"

_FOLDER_LABEL = {_MODE_CREATE: "Location", _MODE_ADOPT: "Corpus folder"}
_FOLDER_HINT = {
    _MODE_CREATE: (
        "Browse to where this corpus should be saved. A subfolder "
        "named after the corpus is created inside it."
    ),
    _MODE_ADOPT: (
        "Browse to the folder that will BE this corpus. Its lancedb/, "
        "corpus.toml, and figures/ are written directly inside it."
    ),
}
_STRUCTURE_CAPTION = {
    _MODE_CREATE: "At the location you pick:",
    _MODE_ADOPT: "In the folder you select:",
}
_STRUCTURE_TREE = {
    _MODE_CREATE: (
        "  <location>/\n"
        "    └── <corpus name>/\n"
        "          ├── lancedb/       (vector store)\n"
        "          ├── corpus.toml    (config file)\n"
        "          └── figures/       (multimodal figures)"
    ),
    _MODE_ADOPT: (
        "  <selected folder>/\n"
        "    ├── lancedb/       (vector store)\n"
        "    ├── corpus.toml    (config file)\n"
        "    └── figures/       (multimodal figures)"
    ),
}


class CreateNewDatasetTab:
    """Storage-only form for registering a new corpus."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.name_field: ft.TextField | None = None
        self.uri_field: ft.TextField | None = None
        self.user_field: ft.TextField | None = None
        self.password_field: ft.TextField | None = None
        self.folder_field: ft.TextField | None = None
        self.location_mode_radio: ft.RadioGroup | None = None
        self.browse_button: ft.Button | None = None
        self.create_button: ft.Button | None = None
        # Right-pane "what gets created" text — kept as refs so the
        # folder-mode radio can rewrite them in place.
        self._structure_caption: ft.Text | None = None
        self._structure_tree: ft.Text | None = None
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)

        self.name_field = ft.TextField(
            label="Corpus name",
            hint_text="e.g. my-papers-2026",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
        )
        self.uri_field = ft.TextField(
            label="Neo4j URI",
            value="neo4j://127.0.0.1:7687",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
        )
        self.user_field = ft.TextField(
            label="Neo4j user",
            value="neo4j",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
        )
        self.password_field = ft.TextField(
            label="Neo4j password",
            password=True,
            can_reveal_password=True,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
        )
        self.folder_field = ft.TextField(
            label=_FOLDER_LABEL[_MODE_CREATE],
            hint_text=_FOLDER_HINT[_MODE_CREATE],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
        )
        # Folder-mode radio — defaults to 'create' (make a named
        # subfolder), matching the folder field's default label + hint.
        self.location_mode_radio = ft.RadioGroup(
            value=_MODE_CREATE,
            on_change=self._on_location_mode_change,
            content=ft.Column(
                spacing=0,
                controls=[
                    ft.Radio(
                        value=_MODE_CREATE,
                        label="Create a new folder named after the corpus",
                    ),
                    ft.Radio(
                        value=_MODE_ADOPT,
                        label="Use the selected folder as the corpus home",
                    ),
                ],
            ),
        )
        # Right-pane structure hint — starts in create-mode wording;
        # `_on_location_mode_change` swaps it to adopt-mode when toggled.
        self._structure_caption = ft.Text(
            _STRUCTURE_CAPTION[_MODE_CREATE],
            size=12,
            color=ft.Colors.GREY_400,
        )
        self._structure_tree = ft.Text(
            _STRUCTURE_TREE[_MODE_CREATE],
            size=11,
            font_family="Consolas",
            color=ft.Colors.GREY_300,
        )
        self.browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_browse_clicked,
        )
        self.create_button = ft.Button(
            content=centered_label("Create corpus"),
            on_click=self.on_create_clicked,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        left_pane = ft.Container(
            expand=_LEFT_FLEX,
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=10,
                controls=[
                    ft.Text(
                        "One Neo4j instance + one LanceDB path per "
                        "corpus. Set the Neo4j DBMS up in Neo4j Desktop "
                        "first, then enter its connection details below.",
                        size=11,
                        color=ft.Colors.GREY_500,
                        italic=True,
                    ),
                    self.name_field,
                    self.uri_field,
                    self.user_field,
                    self.password_field,
                    ft.Text(
                        "Where to put the corpus:",
                        size=12,
                        color=ft.Colors.GREY_400,
                    ),
                    self.location_mode_radio,
                    ft.Row(
                        controls=[self.folder_field, self.browse_button],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        controls=[self.create_button],
                        alignment=ft.MainAxisAlignment.END,
                    ),
                    self.status,
                ],
            ),
        )
        right_pane = ft.Container(
            expand=_RIGHT_FLEX,
            padding=12,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            bgcolor=PANEL_BG,
            border_radius=4,
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=8,
                controls=[
                    ft.Text(
                        "What gets created",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                    ),
                    self._structure_caption,
                    self._structure_tree,
                    ft.Text(
                        "The Neo4j DBMS lives outside — Neo4j Desktop "
                        "manages its data directory itself. Only the "
                        "connection URI is stored per corpus.",
                        size=11,
                        color=ft.Colors.GREY_400,
                    ),
                    ft.Divider(),
                    ft.Text(
                        "After creation",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(
                        "Open Ingest to configure the corpus "
                        "(ontologies, layer flags, extractor, "
                        "thresholds) and run your first ingest.",
                        size=11,
                        color=ft.Colors.GREY_400,
                    ),
                ],
            ),
        )
        body = ft.Row(
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            controls=[left_pane, right_pane],
        )
        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[view_header("Create New Dataset"), body],
        )

    # ----- Browse ----------------------------------------------------------

    async def _on_browse_clicked(self, e: ft.Event) -> None:
        """Open the OS folder picker; drop the choice into the folder field."""
        if self.folder_field is None:
            return
        # Start the picker at the current field value ONLY if the path
        # actually exists on disk. Windows' folder picker fails with
        # `0x80070002 (file not found)` when the initial path doesn't
        # exist yet. Walk up parents until we hit one that exists; pass
        # None if nothing along the chain is present.
        raw = (self.folder_field.value or "").strip()
        initial: str | None = None
        if raw:
            probe = Path(raw)
            while probe != probe.parent and not probe.is_dir():
                probe = probe.parent
            if probe.is_dir():
                initial = str(probe)
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title="Pick a corpus folder",
                initial_directory=initial,
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            if self.status is not None:
                self.status.value = f"folder picker error: {exc}"
            self.app.page.update()
            return
        if not chosen:
            return
        self.folder_field.value = chosen
        self.app.page.update()

    # ----- folder mode -----------------------------------------------------

    def _folder_mode(self) -> str:
        """Current folder mode from the radio: ``create`` | ``adopt``.

        Falls back to ``create`` when the radio isn't built yet, so the
        path derivation stays safe if called before/without a UI.
        """
        if self.location_mode_radio is None:
            return _MODE_CREATE
        return self.location_mode_radio.value or _MODE_CREATE

    def _corpus_folder(self, folder_raw: str, name: str) -> Path:
        """Derive the corpus home from the folder field + the mode radio.

        The single place the folder mode changes anything:

          * ``create`` (default) → ``<folder_raw>/<name>`` — a subfolder
            named after the corpus is made inside the picked parent.
          * ``adopt`` → ``<folder_raw>`` — the picked folder *is* the
            corpus home (adopts an existing folder in place).

        LanceDB / corpus.toml / figures all derive from the returned
        path, so nothing else needs to know which mode is active.
        """
        base = Path(folder_raw)
        return base if self._folder_mode() == _MODE_ADOPT else base / name

    def _on_location_mode_change(self, _e: ft.Event) -> None:
        """Re-label the folder field + right-pane tree for the chosen mode.

        Purely cosmetic — the real path branch lives in `_corpus_folder`.
        """
        mode = self._folder_mode()
        if self.folder_field is not None:
            self.folder_field.label = _FOLDER_LABEL[mode]
            self.folder_field.hint_text = _FOLDER_HINT[mode]
        if self._structure_caption is not None:
            self._structure_caption.value = _STRUCTURE_CAPTION[mode]
        if self._structure_tree is not None:
            self._structure_tree.value = _STRUCTURE_TREE[mode]
        self.app.page.update()

    # ----- validation ------------------------------------------------------

    def _validate(self) -> tuple[bool, str]:
        """Run pre-create sync checks. Returns (ok, error_message).

        Derives the corpus home via `_corpus_folder` (create → a
        `<location>/<name>` subfolder; adopt → the picked folder itself)
        then its internal paths:
          lancedb_path = <corpus folder>/lancedb
          corpus.toml  = <corpus folder>/corpus.toml
        Checks the derived corpus folder doesn't already contain a
        corpus.toml (would clobber an existing corpus) and its lancedb
        subfolder is either nonexistent or empty. These guards are what
        make adopt-mode safe: they refuse a folder that's already a
        corpus while happily adopting one that merely holds documents/.
        """
        if (
            self.name_field is None
            or self.uri_field is None
            or self.user_field is None
            or self.password_field is None
            or self.folder_field is None
        ):
            return False, "form not initialised"
        name = (self.name_field.value or "").strip()
        uri = (self.uri_field.value or "").strip()
        user = (self.user_field.value or "").strip()
        password = self.password_field.value or ""
        folder_raw = (self.folder_field.value or "").strip()

        if not name:
            return False, "corpus name is required"
        if any(c.name == name for c in self.app.gui_config.corpora):
            return False, f"corpus name {name!r} is already registered"
        if not uri:
            return False, "Neo4j URI is required"
        if not user:
            return False, "Neo4j user is required"
        if not password:
            return False, "Neo4j password is required"
        if not folder_raw:
            return False, "corpus folder is required"

        corpus_folder = self._corpus_folder(folder_raw, name)
        toml_path = corpus_folder / "corpus.toml"
        lancedb_path = corpus_folder / "lancedb"
        if toml_path.exists():
            return (
                False,
                f"a corpus already exists at {corpus_folder} "
                f"(corpus.toml present) — pick a different name or folder",
            )
        if lancedb_path.exists() and any(lancedb_path.iterdir()):
            return (
                False,
                f"{lancedb_path} exists and is non-empty — pick a different name or folder",
            )
        return True, ""

    async def _ping_neo4j(self, uri: str, user: str, password: str) -> str | None:
        """Try `RETURN 1` against the URI. Returns None on success, an
        error message on failure.
        """
        # Lazy import so this module stays loadable without neo4j in
        # some contexts (tests, packagers).
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        try:
            async with (
                driver.session() as session,
                await session.begin_transaction() as tx,
            ):
                result = await tx.run("RETURN 1 AS ok")
                row = await result.single()
                if row is None or row.get("ok") != 1:
                    return f"unexpected result from RETURN 1: {row!r}"
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            await driver.close()

    # ----- Create handler --------------------------------------------------

    async def on_create_clicked(self, e: ft.Event) -> None:
        if self.status is None or self.create_button is None:
            return
        # Sync validation first — fail fast on cheap checks.
        ok, msg = self._validate()
        if not ok:
            self.status.value = f"cannot create: {msg}"
            self.app.page.update()
            return

        # Disable the button while the async ping runs so the user
        # doesn't double-submit.
        self.create_button.disabled = True
        self.status.value = "pinging Neo4j…"
        self.app.page.update()

        # Snapshot form values (the fields are guaranteed non-None
        # since _validate passed). Derive the two internal paths from
        # the single Corpus folder field — user picked the location,
        # GUI decides the file structure inside it.
        assert self.name_field is not None
        assert self.uri_field is not None
        assert self.user_field is not None
        assert self.password_field is not None
        assert self.folder_field is not None
        name = self.name_field.value.strip()
        uri = self.uri_field.value.strip()
        user = self.user_field.value.strip()
        password = self.password_field.value
        # Corpus home = the folder field resolved through the mode radio
        # (create → a `<name>` subfolder; adopt → the folder itself). All
        # internal paths hang off that single derived home.
        corpus_folder = self._corpus_folder(self.folder_field.value.strip(), name)
        lancedb_path = corpus_folder / "lancedb"
        toml_path = corpus_folder / "corpus.toml"

        try:
            ping_err = await self._ping_neo4j(uri, user, password)
        except Exception as exc:
            logger.warning("neo4j ping crashed: %r", exc)
            ping_err = f"{type(exc).__name__}: {exc}"
        if ping_err is not None:
            self.status.value = f"Neo4j ping failed: {ping_err}"
            self.create_button.disabled = False
            self.app.page.update()
            return

        # Write corpus.toml with the seed defaults.
        try:
            _write_corpus_toml(toml_path, _seed_corpus_config())
        except Exception as exc:
            logger.warning("write_corpus_config failed: %r", exc)
            self.status.value = f"could not write corpus.toml: {exc}"
            self.create_button.disabled = False
            self.app.page.update()
            return

        # Save password to keyring under the per-corpus namespace.
        try:
            set_corpus_password(name, password)
        except ConfigError as exc:
            self.status.value = f"could not save Neo4j password: {exc}"
            self.create_button.disabled = False
            self.app.page.update()
            return

        # Register in GuiConfig + set active.
        entry = CorpusEntry(
            name=name,
            neo4j_uri=uri,
            neo4j_user=user,
            lancedb_path=lancedb_path,
            corpus_config_path=toml_path,
        )
        self.app.gui_config.corpora = [
            *self.app.gui_config.corpora,
            entry,
        ]
        self.app.gui_config.active_corpus_name = name
        # Mirror the active corpus's storage params to the top-level
        # GuiConfig fields so the existing apply_connection_to_env
        # bridge continues to feed backend Settings correctly.
        self.app.gui_config.neo4j_uri = uri
        self.app.gui_config.neo4j_user = user
        self.app.gui_config.lancedb_path = lancedb_path
        self.app.gui_config.corpus_config_path = toml_path
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save registry: {exc}"
            self.create_button.disabled = False
            self.app.page.update()
            return

        # Bridge + reset caches so subsequent queries hit the new
        # corpus.
        apply_connection_to_env(self.app.gui_config)
        import os

        os.environ["NEO4J_PASSWORD"] = password
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)

        self.status.value = (
            f"corpus {name!r} created + active. Open Select Dataset "
            f"to tune its config (ontologies, layer flags, extractor)."
        )
        self.app.chat_panel.append_system(
            f"created corpus {name!r} — connection saved, corpus.toml written at {toml_path}"
        )
        # Sibling-tab refresh: Select's picker + info card should
        # reflect the new corpus without waiting for the user to click
        # Refresh manually. Reach via `app.library_tab.view.select_tab`
        # — the LibraryView coordinator holds the sibling instance.
        try:
            self.app.library_tab.view.select_tab.on_refresh_clicked(None)
        except Exception as exc:
            logger.warning(
                "sibling refresh after create failed: %r",
                exc,
            )
        # Reset the form so the user can create another if they want.
        self._reset_form()
        self.create_button.disabled = False
        self.app.page.update()

    def _reset_form(self) -> None:
        """Clear the form fields after a successful Create."""
        for field in (
            self.name_field,
            self.password_field,
            self.folder_field,
        ):
            if field is not None:
                field.value = ""


# =============================================================================
# GUI-side corpus.toml writer + seed CorpusConfig for new-corpus creation.
# =============================================================================


def _write_corpus_toml(path: Path, cfg: CorpusConfig) -> None:
    """Serialise a `CorpusConfig` to TOML at `path`.

    Lives on the GUI side because only the GUI writes corpus.toml
    today — the backend's `kg.corpus_config` module is the reader.
    If a CLI ever needs to write corpus.toml, this moves to the
    backend.

    `exclude_none=True` keeps optional-only fields (entities /
    cross_doc / cross_doc_xrefs) out of the file when the user
    hasn't configured them — cleaner minimal TOML at creation time.
    `mode='json'` normalises Path / enum / etc. to strings.

    Writes atomically-ish via `<path>.tmp` + rename so a crash
    mid-write doesn't leave a partial file at the live path. Parent
    directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json", exclude_none=True)
    text = tomlkit.dumps(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def _seed_corpus_config() -> CorpusConfig:
    """Default `CorpusConfig` written to the new corpus's TOML.

    Chunks-only ingest works out of the box; user layers on entities /
    triples / ontologies / cross-doc + picks sub-labels in Select
    Dataset after creation.

    Backend-side defaults:

      - `allowed_types`  : all known sub-labels — user picks one (or
        `(none)`) at ingest time. Narrow via the Corpus section of the
        config editor to restrict the corpus (e.g. paper-only).
      - `layers.chunks`  : True (minimum viable ingest)
      - Other layers off; xrefs = 'none'
      - Ontologies       : {} (user enables in Ingest tab)
      - Entity extractor : not set (user picks when they enable entities)
      - cross_doc / cross_doc_xrefs thresholds : 2 (backend-suggested)
    """
    # Every field defaulted correctly by the model — `allowed_types`
    # gets all sub-labels via its `default_factory`, `layers.chunks` is
    # True with everything else False + xrefs='none', and the optional
    # Cross-doc / Xrefs sub-configs stay None (auto-populated when their
    # layer flag flips on).
    return CorpusConfig()
