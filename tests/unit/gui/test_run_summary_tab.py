"""Tests for the Evaluation Run Summary sub-tab.

`refresh()` renders a run's KPIs + per-case table from a seeded ledger;
an empty ledger shows the empty state. `_ledger` is patched to a tmp DB so
no real eval_output is touched.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import flet as ft

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation._common import ALL_ORIGINS
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab


def _report() -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": "escrt_bootstrap.json",
        "git_commit": "abc12345",
        "prompts_snapshot": {},
        "enabled_groups": ["source", "chunk"],
        "gate_thresholds": {"required_keyword_threshold": 0.5},
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": 0.5},
        "results": [
            {
                "id": "c1",
                "category": "Factual",
                "status": "PASS",
                "errors": [],
                "hit_at_k": 1.0,
                "question": "What is the capital?",
                "expected_output": "Oslo is the capital.",
                "answer": "The capital is Oslo.\n\n## Evidence\n1. Source: x.md",
            },
            {
                "id": "c2",
                "category": "",
                "status": "REVIEW",
                "errors": [],
                "hit_at_k": 0.0,
                "question": "What is the population?",
                "answer": "About 5.5 million.",
            },
        ],
    }


def test_filtered_run_recomputes_over_origins(fake_app: MagicMock, tmp_path):
    """filtered_run keeps only the selected origins and recomputes the summary via
    build_summary; no filter (None / all) returns the stored row + all cases."""
    from knowledge_agent.gui.evaluation._common import filtered_run

    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["results"] = [
        {"id": "m1", "status": "PASS", "origin": "manual", "hit_at_k": 1.0, "errors": []},
        {"id": "m2", "status": "REVIEW", "origin": "manual", "hit_at_k": 0.0, "errors": []},
        {"id": "l1", "status": "PASS", "origin": "llm", "hit_at_k": 1.0, "errors": []},
    ]
    report["summary"] = {"case_count": 3, "pass_count": 2, "pass_rate": 2 / 3}
    led.save_run(report)
    run_id = led.list_runs()[0]["run_id"]
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        # no filter → the stored row + all 3 cases (untouched)
        run_all, cases_all = filtered_run(fake_app, run_id, None)
        assert len(cases_all) == 3
        assert run_all["pass_count"] == 2
        # manual only → 2 cases, recomputed (1 of 2 pass)
        run_m, cases_m = filtered_run(fake_app, run_id, {"manual"})
        assert {c["case_id"] for c in cases_m} == {"m1", "m2"}
        assert run_m["case_count"] == 2
        assert run_m["pass_count"] == 1
        assert run_m["pass_rate"] == 0.5
        assert run_m["git_commit"] == "abc12345"  # provenance survives the overlay


def test_run_summary_builds(fake_app: MagicMock):
    assert (
        RunSummaryTab(
            fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
        ).build()
        is not None
    )


def test_refresh_empty_ledger_shows_empty_state(fake_app: MagicMock, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    tab = RunSummaryTab(
        fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert "No evaluation runs" in tab.body.controls[0].value


def test_refresh_renders_selected_run(fake_app: MagicMock, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert coordinator.selected_run_id == 1  # newest auto-selected
    assert tab.rail.run_dd.options  # run listed in the selector
    assert len(tab.body.controls) > 1  # headline + KPI sections + table (not the empty state)


def test_refresh_with_two_runs_renders_delta_pills(fake_app: MagicMock, tmp_path):
    """With a previous run present, the newest run's cards render delta pills.
    This exercises the pill-rendering path (Flet containers/icons) that a
    single-run render never reaches — a guard against Flet-API breakage there."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())
    r2 = _report()
    r2["summary"] = {"case_count": 2, "pass_count": 2, "pass_rate": 1.0, "avg_hit_at_k": 1.0}
    led.save_run(r2)
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()  # must not raise while building the delta pills
    assert coordinator.selected_run_id == 2  # newest auto-selected
    assert len(tab.body.controls) > 1


def test_delta_pill_direction_and_sign():
    """`_delta` returns a signed vs-previous-run string + whether it improved.
    Higher-is-better metrics improve on an increase; lower-is-better ones
    (latency / tokens) improve on a decrease; a missing previous value (first
    run / not-evaluated) yields no pill."""
    d = RunSummaryTab._delta
    assert d(0.9, 0.8, ".2f", lower_is_better=False) == ("+0.10", True)
    assert d(0.7, 0.8, ".2f", lower_is_better=False) == ("-0.10", False)
    assert d(3.0, 4.0, ".2f", lower_is_better=True) == ("-1.00", True)
    assert d(5.0, 4.0, ".2f", lower_is_better=True) == ("+1.00", False)
    assert d(0.9, None, ".2f", lower_is_better=False) == (None, False)


def test_sort_by_orders_and_flips_on_repeat_click(fake_app: MagicMock, tmp_path):
    """`_sort_by` sets the active column and rebuilds the table; a repeat click
    on the same column flips the direction."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())  # c1: hit_at_k=1.0, c2: hit_at_k=0.0
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    tab._sort_by("hit_at_k")  # ascending → 0.0 before 1.0
    assert [c["case_id"] for c in tab._sorted_cases()] == ["c2", "c1"]
    tab._sort_by("hit_at_k")  # repeat click flips to descending
    assert [c["case_id"] for c in tab._sorted_cases()] == ["c1", "c2"]


def test_sorted_cases_keeps_not_evaluated_last(fake_app: MagicMock):
    """Rows with no value for the sort column (not evaluated) sort last in both
    directions."""
    tab = RunSummaryTab(
        fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )
    tab._cases = [
        {"case_id": "a", "mrr": 0.5},
        {"case_id": "b", "mrr": None},
        {"case_id": "c", "mrr": 0.9},
    ]
    tab._sort_key, tab._sort_asc = "mrr", True
    assert [c["case_id"] for c in tab._sorted_cases()] == ["a", "c", "b"]
    tab._sort_asc = False
    assert [c["case_id"] for c in tab._sorted_cases()] == ["c", "a", "b"]


def test_metric_scores_chart_defaults_all_and_filters(fake_app: MagicMock, tmp_path):
    """The 'Metric Scores by Case' chart offers the 0-1 score columns that have
    data (hit_at_k here), defaults every metric/case selected, and de-selecting
    every metric swaps the chart for a 'select at least one' note."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())  # c1/c2 have hit_at_k values
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    cols = {c for c, _ in tab._chart_metric_cols(tab._cases)}
    assert "hit_at_k" in cols  # a 0-1 score column with data is chartable
    assert tab._chart_metrics == cols  # defaults to all metrics selected
    assert tab._chart_host is not None  # chart rendered into its host
    tab._on_metric_toggle("hit_at_k", False)  # de-select a metric → redraw
    assert "hit_at_k" not in tab._chart_metrics
    for metric in list(tab._chart_metrics):  # de-select the rest
        tab._on_metric_toggle(metric, False)
    assert isinstance(tab._draw_metric_chart(tab._cases), ft.Text)  # guidance note


def test_labeled_box_has_title_and_border():
    """`_labeled_box` — the shared Answer-Detail block shape — renders a bold
    title above a bordered box wrapping the body (used by Quick Stats too)."""
    box = RunSummaryTab._labeled_box("Quick Stats", ft.Text("x"))
    assert box.controls[0].value == "Quick Stats"
    inner = box.controls[1]
    assert inner.border is not None
    assert inner.content.value == "x"


def _text_values(control) -> list[str]:
    """Every `ft.Text`-like value in a control subtree (walks title/content/
    controls) — for asserting a labelled block is present without pinning the
    exact tree shape."""
    out: list[str] = []
    val = getattr(control, "value", None)
    if isinstance(val, str):
        out.append(val)
    for attr in ("title", "content"):
        child = getattr(control, attr, None)
        if child is not None:
            out.extend(_text_values(child))
    for child in getattr(control, "controls", None) or []:
        out.extend(_text_values(child))
    return out


def test_conversation_block_renders_turns():
    raw = json.dumps(
        [
            {"role": "user", "content": "why did the valve fail?"},
            {"role": "assistant", "content": "let me search the corpus"},
        ]
    )
    block = RunSummaryTab._conversation_block(raw)
    assert block is not None
    md = block.controls[1].content  # labeled_box → [label Text, Container(content=Markdown)]
    assert isinstance(md, ft.Markdown)
    assert "**user:** why did the valve fail?" in md.value
    assert "**assistant:** let me search the corpus" in md.value


def test_conversation_block_none_when_empty_or_invalid():
    assert RunSummaryTab._conversation_block(None) is None
    assert RunSummaryTab._conversation_block("[]") is None  # no turns
    assert RunSummaryTab._conversation_block("not json") is None  # unparseable
    # a turn with only whitespace content contributes no line → None
    assert (
        RunSummaryTab._conversation_block(json.dumps([{"role": "user", "content": "  "}])) is None
    )


def test_case_settings_block_renders_knobs():
    """The block shows each stored knob as a label/value cell; a None (unpinned)
    tuning knob reads 'global', and its label is prettified from the field name."""
    raw = json.dumps({"retrieval_mode": "lancedb_only", "top_k": 8, "num_candidates": None})
    block = RunSummaryTab._case_settings_block(raw)
    assert block is not None
    texts = _text_values(block)
    assert "Case Settings" in texts
    assert "Retrieval mode" in texts and "lancedb_only" in texts
    assert "Top k" in texts and "8" in texts
    assert "global" in texts  # None knob → "global"


def test_case_settings_block_none_when_empty_or_invalid():
    assert RunSummaryTab._case_settings_block(None) is None
    assert RunSummaryTab._case_settings_block("{}") is None  # no knobs
    assert RunSummaryTab._case_settings_block("not json") is None  # unparseable


def test_case_details_shows_conversation_only_for_chat_case(fake_app: MagicMock):
    tab = RunSummaryTab(
        fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )
    tab.build()
    cases = [
        {
            "case_id": "c1",
            "status": "PASS",
            "question": "q?",
            "answer": "a",
            "source_conversation": json.dumps([{"role": "user", "content": "hi there"}]),
        },
        {
            "case_id": "c2",
            "status": "PASS",
            "question": "q2?",
            "answer": "a2",
            "source_conversation": "[]",
        },
    ]
    tiles = tab._case_details(cases).controls[1:]  # after the "Case Details" header
    assert any("Conversation (chat)" in s for s in _text_values(tiles[0]))  # chat case
    assert not any("Conversation (chat)" in s for s in _text_values(tiles[1]))  # non-chat


def test_metric_cell_color_is_direction_aware(fake_app: MagicMock):
    """Banded 0-1 metrics color by threshold; a lower-is-better metric
    (hallucination) inverts so a low value reads green; non-score columns
    (tokens) stay plain white; a missing value is grey."""
    from knowledge_agent.evaluation import registry as R

    fmts, directions = R.metric_fmts(), R.metric_directions()
    banded = {
        m.sql_column
        for m in R.METRICS
        if m.sql_column and m.group not in ("tokens", "latency") and m.fmt != "d"
    }
    tab = RunSummaryTab(
        fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )

    def color(val: object, col: str) -> str:
        return tab._metric_cell(val, col, fmts, banded, directions).content.color

    assert color(1.0, "hit_at_k") == ft.Colors.GREEN
    assert color(0.05, "hallucination") == ft.Colors.GREEN  # inverted (lower = better)
    assert color(1234, "agent_total_tokens") == ft.Colors.WHITE  # not a 0-1 score
    assert color(None, "hit_at_k") == ft.Colors.GREY_500  # not evaluated


def test_n_text_reports_cases_behind_a_mean():
    """`_n_text` — 'n = X of Y' for a metric's average (X = cases with a value,
    Y = total). None when the run predates n-tracking so the card omits it."""
    run = {"case_count": 6, "n_hit_at_k": 2}
    assert RunSummaryTab._n_text(run, "hit_at_k") == "n = 2 of 6"
    assert RunSummaryTab._n_text(run, "mrr") is None  # no n_mrr (old run) → omitted
    assert RunSummaryTab._n_text({"n_hit_at_k": 2}, "hit_at_k") is None  # no case_count


def test_refresh_renders_n_line_when_present(fake_app: MagicMock, tmp_path):
    """A run carrying per-metric n renders without error through the _kpi_card
    n branch (the plain _report has no n_ columns, so that branch is otherwise
    never reached)."""
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["summary"]["n_hit_at_k"] = 2
    led.save_run(report)
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert len(tab.body.controls) > 1
