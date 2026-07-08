"""Left column of the Search tab: chat output + ProgressRing +
multiline input + bottom button row.

Owns the chat-side UI and message-rendering helpers (`append_user/
assistant/system`), the busy-state toggle for input/Send/Stop, and
the streaming-bubble helpers used by the synthesizer's astream
output. Cross-cutting actions (Send / Stop / Clear / Save Chat) call
back into `GuiApp`.

Slice 1 doesn't have a corpus selector here — corpus selection lives
in Settings (slice 2). The chat panel surface itself is unchanged
across slices.

Mirrors `research_articles_agent/gui/chat_panel.py` for visual parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    CHAT_EMPTY_PLACEHOLDER,
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class ChatPanel:
    """Left column of the Search tab.

    Persistent controls (`chat_output`, `input_field`, `send_button`,
    `stop_button`) survive view rebuilds — `GuiApp` flips disabled /
    visible / value but doesn't reconstruct them. Keeps cursor position
    + scroll state intact across Send turns.
    """

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.chat_is_empty: bool = True
        # Late-bound — populated by build().
        self.chat_output: ft.Column | None = None
        self.progress_ring: ft.ProgressRing | None = None
        self.input_field: ft.TextField | None = None
        self.send_button: ft.Button | None = None
        self.stop_button: ft.Button | None = None

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        self.chat_output = ft.Column(
            controls=[self._placeholder_text()],
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            expand=True,
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self.progress_ring = ft.ProgressRing(
            visible=False,
            width=18,
            height=18,
            stroke_width=2,
            color=ft.Colors.GREY_500,
        )
        progress_row = ft.Row(
            controls=[self.progress_ring],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        chat_area = ft.Column(
            controls=[self.chat_output, progress_row],
            expand=True,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        output_box = ft.Container(
            content=chat_area,
            bgcolor=PANEL_BG,
            padding=16,
            expand=True,
            border_radius=4,
        )
        # Passive divider between output and input. NOT draggable —
        # Flet 0.85's TextField has a rigid line-based height model
        # (min_lines / max_lines set the actual rendered height; no
        # equivalent of Flutter's `expands: true`), so a draggable
        # vertical splitter would move the divider without resizing
        # the TextField. Kept as a visual separator only.
        splitter = ft.Container(
            content=ft.Divider(height=4, color=FRAME_BORDER_COLOR),
            height=4,
        )

        self.input_field = ft.TextField(
            hint_text=("Ask a question about your corpus ... (Shift+Enter for newline)"),
            multiline=True,
            min_lines=3,
            max_lines=10,
            shift_enter=True,
            on_submit=self.app.on_send,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=False,
        )
        input_row = ft.Row(
            controls=[ft.Container(content=self.input_field, expand=True)],
            expand=False,
        )

        self.send_button = ft.Button(
            content=centered_label("Send"),
            expand=True,
            on_click=self.app.on_send,
        )
        self.stop_button = ft.Button(
            content=centered_label("Stop"),
            expand=True,
            on_click=self.app.on_stop,
            disabled=True,
        )
        button_row = ft.Row(
            controls=[
                ft.Button(
                    content=centered_label("Save"),
                    expand=True,
                    on_click=self.app.on_save_chat,
                ),
                ft.Button(
                    content=centered_label("Clear"),
                    expand=True,
                    on_click=self.app.on_clear,
                ),
                self.stop_button,
                self.send_button,
            ],
            spacing=8,
        )

        inner = ft.Column(
            controls=[output_box, splitter, input_row, button_row],
            expand=True,
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return ft.Container(
            content=inner,
            padding=12,
            bgcolor=PANEL_BG,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=8,
            expand=True,
        )

    def set_busy(self, busy: bool) -> None:
        """Toggle progress + Stop-enabled + input/Send-disabled state."""
        if self.progress_ring is not None:
            self.progress_ring.visible = busy
        if self.send_button is not None:
            self.send_button.disabled = busy
        if self.stop_button is not None:
            # Stop is the inverse of Send: enabled exactly when busy.
            self.stop_button.disabled = not busy
        if self.input_field is not None:
            self.input_field.disabled = busy
        self.app.page.update()

    def reset(self) -> None:
        """Restore the empty-state placeholder; called by Clear."""
        if self.chat_output is not None:
            self.chat_output.controls.clear()
            self.chat_output.controls.append(self._placeholder_text())
        self.chat_is_empty = True

    def get_input_text(self) -> str:
        if self.input_field is None:
            return ""
        return (self.input_field.value or "").strip()

    def clear_input(self) -> None:
        if self.input_field is not None:
            self.input_field.value = ""

    # ----- append helpers ---------------------------------------------------

    def append_user(self, text: str) -> None:
        self._append(self._render_user_message(text))

    def append_assistant(self, text: str) -> None:
        self._append(self._render_assistant_message(text))

    def append_system(self, text: str) -> None:
        self._append(self._render_system_message(text))

    def begin_assistant_stream(self) -> ft.Text:
        """Add a live assistant bubble for streaming output.

        Returns the inner `Text` widget whose `.value` is overwritten on
        each token update (the synthesizer pushes *cumulative* text,
        not deltas, so the consumer just assigns). The bubble looks
        identical to a regular assistant message — it just grows in
        place as text arrives.
        """
        body_text = ft.Text("", size=13, selectable=True)
        container = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "assistant",
                        color=ft.Colors.GREEN_200,
                        size=12,
                        weight=ft.FontWeight.BOLD,
                    ),
                    body_text,
                ],
                spacing=2,
            ),
            padding=ft.Padding.symmetric(vertical=4),
        )
        self._append(container)
        return body_text

    def update_assistant_stream(
        self,
        body_text: ft.Text,
        text: str,
    ) -> None:
        """Overwrite a streaming bubble's text with the latest cumulative value."""
        body_text.value = text
        self.app.page.update()

    def pop_last(self) -> None:
        """Remove the last appended message — used to roll back a failed send."""
        if self.chat_output is not None and self.chat_output.controls:
            self.chat_output.controls.pop()

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _placeholder_text() -> ft.Text:
        return ft.Text(
            CHAT_EMPTY_PLACEHOLDER,
            color=ft.Colors.GREY_500,
            italic=True,
        )

    def _append(self, control: ft.Control) -> None:
        if self.chat_output is None:
            return
        if self.chat_is_empty:
            self.chat_output.controls.clear()
            self.chat_is_empty = False
        self.chat_output.controls.append(control)
        self.app.page.update()

    @staticmethod
    def _render_user_message(text: str) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "you",
                        color=ft.Colors.BLUE_200,
                        size=12,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(text, size=13, selectable=True),
                ],
                spacing=2,
            ),
            padding=ft.Padding.symmetric(vertical=4),
        )

    @staticmethod
    def _render_assistant_message(text: str) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "assistant",
                        color=ft.Colors.GREEN_200,
                        size=12,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(text, size=13, selectable=True),
                ],
                spacing=2,
            ),
            padding=ft.Padding.symmetric(vertical=4),
        )

    @staticmethod
    def _render_system_message(text: str) -> ft.Control:
        return ft.Container(
            content=ft.Text(
                text,
                size=12,
                italic=True,
                color=ft.Colors.GREY_500,
                selectable=True,
            ),
            padding=ft.Padding.symmetric(vertical=2),
        )
