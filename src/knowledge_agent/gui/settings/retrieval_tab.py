"""Settings → Retrieval sub-tab — search-side defaults.

Grouped into visual sections:

  1. Mode — `retrieval_mode` (6 agent-level topologies) +
     `lancedb_search_mode` (LanceDB-internal mode).
  2. Result size — `top_k` (final result count).
  3. Hybrid fusion (RRF) — `num_candidates` (pre-truncation candidate
     pool), `rrf_rank_constant`. Ordering rule: `top_k <= num_candidates`
     (mirrors `Settings._validate_retrieval_windows`).
  4. Diversity (MMR) — `mmr_lambda` (slider).
  5. KG — `kg_max_rows`.
  6. Input mode + synthesis — `input_mode` radio (Conversational /
     Refined query / Direct query / Direct Cypher) + `direct_retrieve`
     (synthesizer skip). The chat-router temperature + model live on the
     LLM tab.

Auto-save pattern (RA-mirror): every control commits its own change.
TextFields validate-then-persist on blur with snap-back on bad input.
Sliders persist on_change_end (after the user lifts off) so we don't
fire a save for every pixel of drag. Dropdowns / Checkboxes persist
on_change.

After a successful save we bridge to env + clear cached factories so
the next search uses the new values without restart.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import check_window_ordering, reset_after_settings_change
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    labeled_field,
    section_divider,
)
from knowledge_agent.gui._widgets.info_icon import section_header
from knowledge_agent.gui._widgets.retrieval_form import (
    LANCE_MODES,
    RetrievalControls,
    apply_gray_out,
    build_input_mode_radios,
    build_mmr_slider,
    build_search_mode_radios,
    mmr_help_text,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_retrieval_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


# Display labels for the 6-mode dropdown. Maps GuiConfig literal value
# to a human-readable label. Order is the dropdown display order;
# `auto` first since it's the default and the most-used option.
_MODE_LABELS: list[tuple[str, str]] = [
    ("auto", "Auto (LLM picks)"),
    ("lancedb_only", "LanceDB only (chunks)"),
    ("neo4j_only", "Neo4j only (KG)"),
    ("lancedb_then_neo4j", "LanceDB then Neo4j (hits, then graph enrichment)"),
    ("neo4j_then_lancedb", "Neo4j then LanceDB (graph filter, then hits)"),
    ("parallel_fused", "Parallel fused (both stores at once, combined in the answer step)"),
]


# 3-tier help text for the Mode section's (i) icon. One icon documents BOTH
# controls the section owns: the agent-level retrieval_mode dropdown and the
# within-store LanceDB search-mode radios (whose parenthetical detail is
# stripped from the radio labels themselves).
_MODE_INFO_STANDARD = (
    "This section has two separate controls.\n\n"
    "Retrieval mode (the dropdown) chooses which store answers your "
    "question:\n"
    "- Auto: the app reads your question and picks one of the five modes "
    "below for you.\n"
    "- LanceDB only: searches the text inside your documents (the chunks). "
    'Best for "what do the documents say about X".\n'
    "- Neo4j only: queries the knowledge graph (the network of authors, "
    "documents, citations and entities) instead of the text. Best for "
    '"who wrote X", "what cites Y", counts and relationships.\n'
    "- LanceDB then Neo4j: finds the most relevant chunks first, then "
    "looks up graph facts about those same documents and adds them.\n"
    "- Neo4j then LanceDB: uses the graph to narrow to a set of documents "
    "first (say, by author or year), then searches chunk text only inside "
    "that set.\n"
    "- Parallel fused: runs the chunk search and the graph query at the "
    "same time, and the answer step combines both.\n\n"
    "LanceDB search mode (the radio buttons) chooses how the chunk search "
    "itself works. It only matters when the chosen mode runs a chunk "
    "search:\n"
    "- Hybrid: combines keyword matching (BM25) with meaning-based vector "
    "search. The all-round default.\n"
    "- FTS: keyword matching only. Best when exact words matter (names, "
    "codes, precise phrases).\n"
    "- Vector: meaning-based only. Best when your wording differs from the "
    "documents but the concept is the same.\n\n"
    "Controls the current mode does not use are greyed out."
)
_MODE_INFO_BEGINNER = (
    "Two questions decide how a search runs.\n\n"
    "First, where should the app look? That is the Retrieval mode "
    "dropdown:\n"
    "- Auto: let the app choose for you. Use this if you are unsure.\n"
    "- LanceDB only: read the actual words inside your documents. Use it "
    'for "what do the documents say about ...".\n'
    "- Neo4j only: look at the map of who wrote what and what links to "
    "what, rather than the text. Use it for "
    '"who wrote ...", "what cites ...", or counting things.\n'
    "- LanceDB then Neo4j: find the text first, then add facts about those "
    "documents (such as their authors or citations).\n"
    "- Neo4j then LanceDB: first pick documents by a fact (like an "
    "author), then read the text only inside those.\n"
    "- Parallel fused: do both at once and blend the result. Good for big, "
    "open-ended questions.\n\n"
    "Second, how should it search the text? That is the LanceDB search "
    "mode buttons (they only matter when the app reads document text):\n"
    "- Hybrid: matches both exact words and meaning. Leave it here if "
    "unsure.\n"
    "- FTS: exact words only. Good when a specific term or name must "
    "appear.\n"
    "- Vector: meaning only. Good when you describe an idea in your own "
    "words.\n\n"
    "Starting out, Auto with Hybrid is a safe pairing."
)
_MODE_INFO_TECHNICAL = (
    "Agent-level mode maps to Settings.default_retrieval_mode (env "
    "DEFAULT_RETRIEVAL_MODE) and is also passed per-search as a state "
    "override that wins over the saved default. It selects the LangGraph "
    "topology:\n"
    "- auto: an LLM mode_classifier reads the query and picks one of the "
    "five concrete modes (fail-soft to lancedb_only on error).\n"
    "- lancedb_only: query_builder, then lancedb_retriever, then "
    "synthesizer.\n"
    "- neo4j_only: cypher_builder, then neo4j_retriever, then "
    "synthesizer.\n"
    "- lancedb_then_neo4j: the chunk hits are fed into cypher_builder so "
    "its Cypher can correlate by doc_id.\n"
    "- neo4j_then_lancedb: the doc_ids from the graph rows become a "
    '"doc_id IN (...)" filter on the chunk search (unfiltered fallback '
    "when none).\n"
    "- parallel_fused: both legs fan out and the synthesizer LLM merges "
    "the two evidence lists. There is no cross-store rank fusion.\n\n"
    "Within-store mode maps to Settings.lancedb_search_mode (env "
    "LANCEDB_SEARCH_MODE):\n"
    "- hybrid: BM25 and cosine-vector legs fused by LanceDB's RRF "
    "reranker.\n"
    "- fts: BM25 only (MMR does not apply).\n"
    "- vector: cosine kNN only.\n\n"
    "Direct Cypher input mode forces the store to neo4j_only, overriding "
    "this dropdown. Unused knobs grey out via apply_gray_out."
)


# 3-tier help text for the other four Retrieval sections' (i) icons. Same rule
# as the Mode block above: every control the section owns is explained in all
# three tiers (standard, beginner, technical); only the depth changes.
_RESULT_SIZE_INFO_STANDARD = (
    "top_k is how many document chunks a search returns: the final result "
    "count.\n\n"
    "The search gathers a larger pool of candidate chunks, ranks them, and "
    "keeps the best top_k. It caps document chunks only; graph rows have "
    "their own limit (see Knowledge graph). The allowed range is 1 to 50, "
    "and top_k must not exceed num_candidates (the form reverts the value "
    "if you break that).\n\n"
    'It has no effect in "Neo4j only" mode, which returns graph rows, not '
    "chunks."
)
_RESULT_SIZE_INFO_BEGINNER = (
    "This is simply how many results come back. Around 5 is a good start: "
    "enough to answer most questions without burying you in text. Raise it "
    "to have the assistant weigh more sources, lower it for short, focused "
    "answers. The most you can set is 50."
)
_RESULT_SIZE_INFO_TECHNICAL = (
    "Maps to Settings.top_k (env TOP_K); also sent per-search in the graph "
    "state, where it overrides the saved default. Constraints: 1 to 50, "
    "and top_k must be less than or equal to num_candidates (the Retrieval "
    "tab checks this itself and reverts an edit that breaks it). In hybrid "
    "and vector search the ranked pool is cut, or MMR-reranked, to top_k; "
    "FTS limits straight to top_k. It is not greyed in Neo4j-only mode but "
    "has no effect there (no chunk leg runs)."
)
_RRF_INFO_STANDARD = (
    "Two knobs that shape the chunk search before it returns your top_k "
    "results.\n\n"
    "- num_candidates: how many rows the search pulls into its working "
    "pool before trimming down to top_k. A bigger pool means better recall "
    "and better fusion, at a little more cost. It applies to Hybrid and "
    "Vector search, and must be at least top_k (the form reverts an edit "
    "that breaks this).\n"
    "- RRF rank constant k: used only by Hybrid. When Hybrid blends its "
    "keyword (BM25) and vector results, each hit scores 1/(k + rank). A "
    "lower k lets the very top hits dominate; a higher k evens out the "
    "difference between ranks."
)
_RRF_INFO_BEGINNER = (
    'These fine-tune the "Hybrid" chunk search, and you can usually leave '
    "them alone.\n\n"
    "- num_candidates: how wide a net the search casts before picking the "
    "best few. Larger can find slightly better results but is a touch "
    "slower. The default (100) suits most corpora.\n"
    "- RRF rank constant: how strongly the very top matches are favoured "
    "when Hybrid blends keyword and meaning results. The default (60) is a "
    "sensible balance. Lower favours the top hits more; higher treats more "
    "results evenly."
)
_RRF_INFO_TECHNICAL = (
    "num_candidates maps to Settings.num_candidates (env NUM_CANDIDATES, "
    "default 100); the RRF constant to Settings.rrf_rank_constant (env "
    "RRF_RANK_CONSTANT, default 60). In Hybrid the BM25 and cosine-vector "
    "legs each fetch num_candidates rows and LanceDB's RRF reranker fuses "
    "them with score 1/(k + rank); the fused pool is then cut to top_k, or "
    "MMR-reranked. Vector search also fetches num_candidates but ignores "
    "k. FTS uses neither and limits straight to top_k, so num_candidates "
    "has no effect in FTS (it stays editable there; the RRF constant greys "
    "out in both FTS and Vector). From the GUI these two are saved "
    "settings, applied on the next search, not per-search overrides."
)
_MMR_INFO_STANDARD = (
    "MMR (Maximal Marginal Relevance) re-ranks the chunk pool to cut "
    "near-duplicates, so the results cover more ground instead of "
    "repeating the same point.\n\n"
    "- Apply MMR diversity re-rank (checkbox): turns the re-rank on or "
    "off.\n"
    "- MMR lambda (slider): the balance between relevance and variety. "
    "Higher favours relevance (results closest to your query); lower "
    "favours diversity (results that differ from each other).\n\n"
    "MMR needs vectors, so it applies to Hybrid and Vector search, not FTS."
)
_MMR_INFO_BEGINNER = (
    "Turn this on when your results feel repetitive.\n\n"
    "- Apply MMR diversity re-rank: when several top chunks say almost the "
    "same thing, tick this to spread the results across different points. "
    "Off by default.\n"
    "- MMR lambda: slide toward the right for the most on-topic results, "
    "or toward the left for more variety. The middle is a sensible "
    "balance.\n\n"
    "It only works with the Hybrid or Vector search modes (it needs "
    "meaning-based vectors), so it is unavailable for FTS."
)
_MMR_INFO_TECHNICAL = (
    "use_mmr maps to Settings.default_use_mmr (env DEFAULT_USE_MMR, "
    "default off); mmr_lambda to Settings.mmr_lambda (env MMR_LAMBDA, 0.0 "
    "to 1.0, default 0.6). When on, the num_candidates pool is re-ranked "
    "Python-side down to top_k by score = lambda * sim(query, c) minus "
    "(1 - lambda) * max sim(c, already-picked), cosine on each row's "
    "stored embedding. lambda 1.0 is pure relevance, 0.0 is pure "
    "diversity. FTS ignores use_mmr (no query vector); a pool row missing "
    "its embedding falls back to plain relevance order. The checkbox greys "
    "out in FTS and Neo4j-only; the slider greys out whenever MMR is off. "
    "Both are saved settings applied on the next search, not per-search "
    "overrides."
)
_KG_INFO_STANDARD = (
    "kg_max_rows is the most rows a single knowledge-graph query may "
    "return.\n\n"
    "It is a hard cap on graph results (authors, citations, entities and "
    "other structured rows). It is separate from top_k, which caps "
    "document chunks; graph queries often return more rows, for example "
    "the 50 citations of a paper.\n\n"
    'It has no effect in "LanceDB only" mode, which does not query the '
    "graph."
)
_KG_INFO_BEGINNER = (
    "This limits how many rows the graph side hands back at once. The "
    "default (50) is plenty for most questions. Raise it when you ask for "
    "long lists (say, every author across many papers) and want to see "
    'more; lower it to keep answers tight. It does nothing in "LanceDB '
    'only" mode, which never touches the graph.'
)
_KG_INFO_TECHNICAL = (
    "Maps to Settings.kg_max_rows (env KG_MAX_ROWS, default 50). The Neo4j "
    'retriever wraps the generated Cypher as "CALL { <cypher> } RETURN * '
    "LIMIT kg_max_rows\", so Neo4j enforces the cap even when the LLM's "
    "own query forgets a LIMIT or sets a larger one. Distinct from top_k. "
    'Greys out in "LanceDB only" (no graph leg runs), and is a saved '
    "setting applied on the next search."
)


_INPUT_INFO_STANDARD = (
    "This section has two controls: how your typed input becomes a search, "
    "and whether the app writes an answer or just shows the raw results.\n\n"
    "Input mode (the radios) decides how your text is turned into a query:\n"
    "- Conversational: the chat assistant reads the conversation, decides "
    "when to search, and turns your intent into a query for you. The normal "
    "chat experience.\n"
    "- Refined query: no chat assistant, but the query builder still "
    "rewrites your text into a search-friendly query before searching.\n"
    "- Direct query: your exact text is the search query, with no "
    "rewriting.\n"
    "- Direct Cypher: your text is run as a read-only Cypher query against "
    "the knowledge graph. This forces the search to Neo4j only, whatever the "
    "Mode dropdown says.\n\n"
    "Direct retrieval (the checkbox) decides what comes back:\n"
    "- Off: the app writes a synthesized answer with citations (the normal "
    "result).\n"
    "- On: the app skips the answer-writing step and returns the raw "
    "retrieved chunks and graph rows so you can read the sources yourself. "
    "Works with every retrieval mode."
)
_INPUT_INFO_BEGINNER = (
    "Leave Input mode on Conversational and Direct retrieval off, and it "
    "works like a normal chat: you type a question in plain words, and the "
    "app figures out how and when to search and writes you an answer.\n\n"
    "The other input modes are shortcuts for advanced users:\n"
    "- Refined query and Direct query give you more control over the exact "
    "search words.\n"
    "- Direct Cypher is for people who want to write graph queries by "
    "hand.\n\n"
    "Turn Direct retrieval on only if you want to see the raw sources the "
    "app found, without a written answer. That is mainly for checking search "
    "quality."
)
_INPUT_INFO_TECHNICAL = (
    "Input mode is a GUI-only routing choice (GuiConfig.input_mode); it is "
    "not a backend setting. It maps at query time to graph knobs:\n"
    "- conversational: runs the chat router node, which gates retrieval "
    "(ready_to_retrieve) and distills a search_query, then the graph runs. "
    "The only mode that uses the router.\n"
    "- refined: skips the router; query_builder rewrites the text "
    "(skip_query_builder=False).\n"
    "- direct_query: skips the router and query_builder; the raw text is the "
    "query (skip_query_builder=True).\n"
    "- direct_cypher: the text is passed as user_cypher and run verbatim; "
    "store_forced_by_mode pins retrieval_mode to neo4j_only, and the "
    "read-only rails (is_cypher_read_only, LIMIT wrap, read session) reject "
    "writes.\n\n"
    "Direct retrieval maps to direct_retrieval (env DIRECT_RETRIEVAL plus a "
    "per-invocation state field, which wins from the GUI). When on, the "
    "synthesizer node is skipped: the answer is empty and the retrieved "
    "chunks and KG rows are returned as sources. It applies to every "
    "topology, since the synthesizer is the single convergence node."
)


class RetrievalTab:
    """Retrieval defaults — auto-save fields with an RRF ordering check."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.top_k_field: ft.TextField | None = None
        self.mode_dropdown: ft.Dropdown | None = None
        self.lancedb_mode_radio: ft.RadioGroup | None = None
        self._lancedb_radios: list[ft.Radio] = []
        self._lancedb_mode_box: ft.Container | None = None
        self.num_candidates_field: ft.TextField | None = None
        self.rrf_constant_field: ft.TextField | None = None
        self.mmr_help_text: ft.Text | None = None
        self.use_mmr_checkbox: ft.Checkbox | None = None
        self.mmr_lambda_slider: ft.Slider | None = None
        self.mmr_lambda_value_text: ft.Text | None = None
        self.kg_max_rows_field: ft.TextField | None = None
        self.input_mode_radio: ft.RadioGroup | None = None
        self.direct_retrieve_checkbox: ft.Checkbox | None = None
        self._create_controls()
        # Initial gray-out based on the loaded mode + lancedb mode + MMR state.
        self._sync_enabled_state()
        # Grey the Mode dropdown when Direct Cypher input mode is active.
        self._sync_mode_override_cue()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)
        cfg = self.app.gui_config

        # Mode dropdowns.
        # Flet 0.85 `Dropdown` uses `on_select` (not `on_change`); the
        # event fires after the user picks an option from the menu.
        # 6-option agent-level mode stays a dropdown — too many for radios
        # in a narrow panel. 3-option LanceDB mode uses radios (all
        # options visible at once, one click to switch).
        self.mode_dropdown = ft.Dropdown(
            value=cfg.retrieval_mode,
            options=[ft.DropdownOption(key=k, text=lbl) for k, lbl in _MODE_LABELS],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_select=self.on_mode_changed,
        )
        # Radios + slider come from the shared builders, so the eval Dataset
        # form renders the identical widgets (single source, no look-alikes).
        self._lancedb_radios, self.lancedb_mode_radio, self._lancedb_mode_box = (
            build_search_mode_radios(cfg.lancedb_search_mode, self.on_lancedb_mode_changed)
        )

        # Integer fields.
        self.top_k_field = _int_field(
            cfg.top_k,
            self.on_top_k_blur,
        )
        self.num_candidates_field = _int_field(
            cfg.num_candidates,
            self.on_num_candidates_blur,
        )
        self.rrf_constant_field = _int_field(
            cfg.rrf_rank_constant,
            self.on_rrf_constant_blur,
        )
        self.kg_max_rows_field = _int_field(
            cfg.kg_max_rows,
            self.on_kg_max_rows_blur,
        )

        # MMR enable checkbox — when off, slider+multiplier are grayed.
        # When LanceDB mode is FTS, the checkbox itself is grayed (MMR
        # is meaningless there).
        self.use_mmr_checkbox = ft.Checkbox(
            label="Apply MMR diversity re-rank",
            value=cfg.use_mmr,
            on_change=self.on_use_mmr_changed,
        )
        # Sliders for 0-1 floats. on_change updates the inline value
        # display; on_change_end persists (avoid one save per pixel).
        self.mmr_help_text = ft.Text(
            "lambda 1.0 is pure relevance; lambda 0.0 is pure diversity.",
            size=12,
            color=ft.Colors.GREY_500,
            italic=True,
        )
        self.mmr_lambda_slider, self.mmr_lambda_value_text = build_mmr_slider(
            cfg.mmr_lambda,
            on_change=self._on_mmr_lambda_slide,
            on_change_end=self.on_mmr_lambda_committed,
        )
        self.input_mode_radio = ft.RadioGroup(
            value=cfg.input_mode,
            on_change=self.on_input_mode_changed,
            content=ft.Column(
                spacing=2,
                controls=[
                    # Conversational is chat-only (the GUI chat router). It lives
                    # here, NOT in the shared builder — the eval Dataset form uses
                    # that builder too and has no router. The other three modes
                    # come from `build_input_mode_radios` so the two forms can't
                    # drift on options, labels, or wiring.
                    ft.Radio(
                        value="conversational",
                        label=(
                            "Conversational — the chat router refines intent "
                            "and decides when to search"
                        ),
                    ),
                    *build_input_mode_radios(),
                ],
            ),
        )

        self.direct_retrieve_checkbox = ft.Checkbox(
            label=("Direct retrieval — skip synthesizer, show raw chunks / rows"),
            value=cfg.direct_retrieve,
            on_change=self.on_direct_retrieve_changed,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("Retrieval"),
                # ---- Mode ----------------------------------------------
                section_header(
                    self.app,
                    "Mode",
                    _MODE_INFO_STANDARD,
                    beginner=_MODE_INFO_BEGINNER,
                    technical=_MODE_INFO_TECHNICAL,
                ),
                labeled_field("Retrieval mode (agent-level)", self.mode_dropdown),
                self._lancedb_mode_box,
                section_divider(),
                # ---- Result size ---------------------------------------
                section_header(
                    self.app,
                    "Result size",
                    _RESULT_SIZE_INFO_STANDARD,
                    beginner=_RESULT_SIZE_INFO_BEGINNER,
                    technical=_RESULT_SIZE_INFO_TECHNICAL,
                ),
                labeled_field("top_k (final result count, 1 to 50)", self.top_k_field),
                section_divider(),
                # ---- Hybrid fusion (RRF) -------------------------------
                section_header(
                    self.app,
                    "Hybrid fusion (RRF)",
                    _RRF_INFO_STANDARD,
                    beginner=_RRF_INFO_BEGINNER,
                    technical=_RRF_INFO_TECHNICAL,
                ),
                labeled_field(
                    "num_candidates (candidate pool before top_k)",
                    self.num_candidates_field,
                ),
                labeled_field("RRF rank constant k (1/(k+rank))", self.rrf_constant_field),
                section_divider(),
                # ---- Diversity (MMR) -----------------------------------
                section_header(
                    self.app,
                    "Diversity (MMR)",
                    _MMR_INFO_STANDARD,
                    beginner=_MMR_INFO_BEGINNER,
                    technical=_MMR_INFO_TECHNICAL,
                ),
                self.mmr_help_text,
                self.use_mmr_checkbox,
                _slider_row(
                    "MMR lambda",
                    self.mmr_lambda_slider,
                    self.mmr_lambda_value_text,
                ),
                section_divider(),
                # ---- Knowledge graph -----------------------------------
                section_header(
                    self.app,
                    "Knowledge graph",
                    _KG_INFO_STANDARD,
                    beginner=_KG_INFO_BEGINNER,
                    technical=_KG_INFO_TECHNICAL,
                ),
                labeled_field("kg_max_rows (cap on Neo4j rows per query)", self.kg_max_rows_field),
                section_divider(),
                # ---- Input mode + synthesis ----------------------------
                section_header(
                    self.app,
                    "Input mode",
                    _INPUT_INFO_STANDARD,
                    beginner=_INPUT_INFO_BEGINNER,
                    technical=_INPUT_INFO_TECHNICAL,
                ),
                self.input_mode_radio,
                self.direct_retrieve_checkbox,
                # ---- Shared status text --------------------------------
                self.status,
            ],
        )

    # ----- save core --------------------------------------------------------

    def _commit(self, label: str) -> bool:
        """Persist + bridge + reset caches. Returns True on success.

        Status text + chat append handle user feedback. Bridging
        re-runs apply_retrieval_to_env on the WHOLE GuiConfig so any
        valid in-memory edit reaches env (saves us per-field bridge
        functions).
        """
        if self.status is None:
            return False
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return False
        apply_retrieval_to_env(self.app.gui_config)
        try:
            reset_after_settings_change()
        except Exception as exc:
            logger.warning("reset_after_settings_change failed: %r", exc)
        self.status.value = f"saved {label}"
        self.app.page.update()
        return True

    # ----- Dropdown handlers -----------------------------------------------

    def on_mode_changed(self, e: ft.Event) -> None:
        if self.mode_dropdown is None:
            return
        new_value = self.mode_dropdown.value
        if new_value == self.app.gui_config.retrieval_mode:
            return
        previous = self.app.gui_config.retrieval_mode
        self.app.gui_config.retrieval_mode = new_value  # type: ignore[assignment]
        if not self._commit(f"retrieval mode: {new_value}"):
            self.app.gui_config.retrieval_mode = previous
            self.mode_dropdown.value = previous
        # Mode change flips which legs run → re-gray the LanceDB / KG blocks.
        self._sync_enabled_state()
        self.app.page.update()

    def on_lancedb_mode_changed(self, e: ft.Event) -> None:
        if self.lancedb_mode_radio is None:
            return
        new_value = self.lancedb_mode_radio.value
        if new_value == self.app.gui_config.lancedb_search_mode:
            return
        previous = self.app.gui_config.lancedb_search_mode
        self.app.gui_config.lancedb_search_mode = new_value  # type: ignore[assignment]
        if not self._commit(f"LanceDB search mode: {new_value}"):
            self.app.gui_config.lancedb_search_mode = previous
            self.lancedb_mode_radio.value = previous
        # MMR enable/disable state depends on the LanceDB mode (FTS has
        # no vectors → MMR doesn't apply). Sync after every mode change.
        self._sync_enabled_state()
        self.app.page.update()

    def _sync_mode_override_cue(self) -> None:
        """Grey the Mode dropdown when Direct Cypher input mode is selected.

        Direct Cypher pins the store to neo4j_only (`store_forced_by_mode`)
        regardless of the dropdown, so graying it out signals that the
        dropdown's value is overridden for this input mode."""
        if self.mode_dropdown is not None:
            self.mode_dropdown.disabled = self.app.gui_config.input_mode == "direct_cypher"

    def _sync_enabled_state(self) -> None:
        """Gray out every control the current mode/toggles can't use, via the
        shared `apply_gray_out` (the single source both this tab and the eval
        form use), then refresh the MMR help line."""
        cfg = self.app.gui_config
        apply_gray_out(
            RetrievalControls(
                lancedb_mode_box=self._lancedb_mode_box,
                lancedb_radios=self._lancedb_radios,
                lancedb_mode_control=self.lancedb_mode_radio,
                num_candidates_field=self.num_candidates_field,
                rrf_constant_field=self.rrf_constant_field,
                use_mmr_checkbox=self.use_mmr_checkbox,
                mmr_lambda_control=self.mmr_lambda_slider,
                kg_max_rows_field=self.kg_max_rows_field,
            ),
            retrieval_mode=cfg.retrieval_mode,
            lancedb_search_mode=cfg.lancedb_search_mode,
            use_mmr=cfg.use_mmr,
        )
        if self.mmr_help_text is not None:
            self.mmr_help_text.value = mmr_help_text(
                lance_on=cfg.retrieval_mode in LANCE_MODES,
                is_fts=cfg.lancedb_search_mode == "fts",
                use_mmr=cfg.use_mmr,
            )

    def on_use_mmr_changed(self, e: ft.Event) -> None:
        """Persist use_mmr; re-sync slider/multiplier gray-out."""
        if self.use_mmr_checkbox is None:
            return
        previous = self.app.gui_config.use_mmr
        self.app.gui_config.use_mmr = bool(self.use_mmr_checkbox.value)
        if not self._commit("MMR on" if self.app.gui_config.use_mmr else "MMR off"):
            self.app.gui_config.use_mmr = previous
            self.use_mmr_checkbox.value = previous
        self._sync_enabled_state()
        self.app.page.update()

    # ----- Integer TextField handlers --------------------------------------

    def on_top_k_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.top_k_field,
            new_value_parser=lambda raw: max(1, min(50, int(raw))),
            current=self.app.gui_config.top_k,
            label="top_k",
            apply=lambda v: setattr(self.app.gui_config, "top_k", v),
            validate=lambda: check_window_ordering(
                self.app.gui_config.top_k, self.app.gui_config.num_candidates
            ),
        )

    def on_num_candidates_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.num_candidates_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.num_candidates,
            label="num_candidates",
            apply=lambda v: setattr(self.app.gui_config, "num_candidates", v),
            validate=lambda: check_window_ordering(
                self.app.gui_config.top_k, self.app.gui_config.num_candidates
            ),
        )

    def on_rrf_constant_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.rrf_constant_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.rrf_rank_constant,
            label="RRF rank constant",
            apply=lambda v: setattr(
                self.app.gui_config,
                "rrf_rank_constant",
                v,
            ),
            validate=None,
        )

    def on_kg_max_rows_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.kg_max_rows_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.kg_max_rows,
            label="kg_max_rows",
            apply=lambda v: setattr(self.app.gui_config, "kg_max_rows", v),
            validate=None,
        )

    def _handle_int(
        self,
        *,
        field: ft.TextField | None,
        new_value_parser,
        current: int,
        label: str,
        apply,
        validate,
    ) -> None:
        """Shared blur logic: parse → validate → apply → commit (rollback
        on failure or invalid input).

        `new_value_parser(raw_str) -> int` clamps to the field's valid
        range. `apply(value)` writes to GuiConfig. `validate` is an
        optional callable that takes no args + returns an error
        message (str) or None — checked AFTER apply, before persist.
        """
        if field is None or self.status is None:
            return
        raw = (field.value or "").strip()
        try:
            new_value = new_value_parser(raw)
        except (ValueError, TypeError):
            field.value = str(current)
            self.status.value = f"{label} must be a positive integer"
            self.app.page.update()
            return
        if new_value == current:
            field.value = str(new_value)  # normalize
            self.app.page.update()
            return
        apply(new_value)
        if validate is not None:
            err = validate()
            if err is not None:
                apply(current)
                field.value = str(current)
                self.status.value = err
                self.app.page.update()
                return
        if not self._commit(f"{label}: {new_value}"):
            apply(current)
            field.value = str(current)
            self.app.page.update()
            return
        field.value = str(new_value)
        self.app.page.update()

    # ----- Slider handlers --------------------------------------------------

    def _on_mmr_lambda_slide(self, e: ft.Event) -> None:
        """Drag-in-progress: the shared builder already synced the value text, so
        just flush it to the page."""
        self.app.page.update()

    def on_mmr_lambda_committed(self, e: ft.Event) -> None:
        if self.mmr_lambda_slider is None:
            return
        new_value = float(self.mmr_lambda_slider.value)
        if abs(new_value - self.app.gui_config.mmr_lambda) < 1e-6:
            return
        previous = self.app.gui_config.mmr_lambda
        self.app.gui_config.mmr_lambda = new_value
        if not self._commit(f"MMR lambda: {new_value:.2f}"):
            self.app.gui_config.mmr_lambda = previous
            self.mmr_lambda_slider.value = previous
            if self.mmr_lambda_value_text is not None:
                self.mmr_lambda_value_text.value = _fmt_float(previous)
            self.app.page.update()

    def on_input_mode_changed(self, e: ft.Event) -> None:
        """Persist the input mode (Conversational / Refined query / Direct query
        / Direct Cypher), then re-sync the Mode-dropdown override cue."""
        if self.input_mode_radio is None:
            return
        previous = self.app.gui_config.input_mode
        self.app.gui_config.input_mode = self.input_mode_radio.value
        if not self._commit(f"input mode: {self.app.gui_config.input_mode}"):
            self.app.gui_config.input_mode = previous
            self.input_mode_radio.value = previous
        # Direct Cypher forces the store to neo4j_only, so grey the Mode dropdown
        # to signal the override (and un-grey it on any other input mode).
        self._sync_mode_override_cue()
        self.app.page.update()

    # ----- Checkbox handlers -----------------------------------------------

    def on_direct_retrieve_changed(self, e: ft.Event) -> None:
        if self.direct_retrieve_checkbox is None:
            return
        previous = self.app.gui_config.direct_retrieve
        self.app.gui_config.direct_retrieve = bool(self.direct_retrieve_checkbox.value)
        if not self._commit(
            "direct retrieval on (synthesizer skipped)"
            if self.app.gui_config.direct_retrieve
            else "direct retrieval off"
        ):
            self.app.gui_config.direct_retrieve = previous
            self.direct_retrieve_checkbox.value = previous
            self.app.page.update()


# =============================================================================
# Module-local widget helpers.
# =============================================================================


def _int_field(
    value: int,
    on_blur,
) -> ft.TextField:
    """Standard integer-input TextField using the panel's style. The caption
    is added at the build site via `labeled_field`."""
    return ft.TextField(
        value=str(value),
        border=ft.InputBorder.OUTLINE,
        border_color=FRAME_BORDER_COLOR,
        bgcolor=PANEL_BG,
        on_blur=on_blur,
    )


def _slider_row(
    label: str,
    slider: ft.Slider,
    value_text: ft.Text,
) -> ft.Control:
    """Caption + slider (expanding) + numeric value display, in the shared
    `labeled_field` style (caption hugs its text, value trails)."""
    return labeled_field(label, slider, trailing=value_text)


def _fmt_float(value: float | None) -> str:
    """Two-decimal display for slider value (handles None defensively)."""
    if value is None:
        return "0.00"
    return f"{float(value):.2f}"
