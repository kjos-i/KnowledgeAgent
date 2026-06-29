"""Shared helpers for `scripts/smoke_install_*.py`.

Lifecycle pattern (per [[feedback-smoke-test-cleanup]]):

  1. Print current state (already installed? plan summary?)
  2. Confirm install with the user.
  3. Run install.
  4. Make ONE small invocation to confirm the adapter works.
  5. Pause for inspection.
  6. Offer uninstall (default: keep).

`confirm` returns True on Enter (default Yes) unless the user types
"n" or "N". `confirm_no_default` defaults to No (used for the
uninstall-at-cleanup prompt — keeping the install is the safer
default since the user just paid the install cost).
"""

from __future__ import annotations

import sys
from typing import Any


def header(text: str) -> None:
    """One-line section banner."""
    print(f"\n=== {text} ===")


def confirm(prompt: str) -> bool:
    """Yes/No prompt with Yes as the default (Enter = yes)."""
    reply = input(f"{prompt} [Y/n] ").strip().lower()
    return reply in ("", "y", "yes")


def confirm_no_default(prompt: str) -> bool:
    """Yes/No prompt with No as the default (Enter = no)."""
    reply = input(f"{prompt} [y/N] ").strip().lower()
    return reply in ("y", "yes")


def print_plan(label: str, plan: Any) -> None:
    """Render a lifecycle plan dataclass using its `summary` property."""
    header(f"PLAN: {label}")
    print(plan.summary)


def print_result(label: str, result: Any) -> None:
    """Render a lifecycle result dataclass field-by-field."""
    header(f"RESULT: {label}")
    for field, value in vars(result).items():
        # Truncate pip output to keep the smoke log readable; full
        # output is visible in `pip`'s own stdout during execute.
        if field.endswith("output") and isinstance(value, str) and len(value) > 500:
            print(f"  {field}: <{len(value)} chars, truncated>")
            print(f"    {value[:200]}...")
        else:
            print(f"  {field}: {value!r}")


def bail_if_not_confirmed(prompt: str) -> None:
    """Abort the smoke if the user doesn't confirm."""
    if not confirm(prompt):
        print("Aborted.")
        sys.exit(0)
