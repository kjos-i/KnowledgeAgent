"""Smoke test for the entity-extractor install lifecycle.

Exercises the full plan/execute pattern against the GLiNER adapter
(general-purpose zero-shot NER, ~1.1 GB safetensors). GLiNER chosen
because:

  - Apache-2.0 + safetensors only → install dialog clean (no pickle
    warning to wade through)
  - General domain tags → smoke runs regardless of corpus focus
  - First HF-cache adapter ever shipped → exercises the most common
    cross-vendor pattern (pip extra + HF model cache + lazy load)

Flow:

  1. `install_extractor_plan("gliner")` — prints provenance (publisher,
     license, pinned commit SHA, safetensors flag,
     trust_remote_code flag, domain tags, emitted labels, cross-
     link surface against installed ontologies).
  2. User confirms; `install_extractor_execute(plan)` runs `pip install`
     for the `entities-gliner` extra. Model weights download lazily
     on first inference, not at install time.
  3. Run one inference call to confirm the adapter is wired correctly.
  4. Pause for inspection.
  5. Optional `uninstall_extractor_execute(plan)` to remove the pip
     package. HF cache files persist on disk for future re-installs.

Counterpart automation:
  - tests/unit/entity_extractors/test_extractor_lifecycle.py covers
    the plan/execute logic with mocked pip + mocked is_installed.
  - tests/integration/test_entity_extractors_real_models.py runs the
    real adapter inference (gated by @slow); this smoke is the
    human-supervised version of that path.

Run from the project root:
    python scripts/smoke_install_extractor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _install_smoke_lib import (  # noqa: E402
    bail_if_not_confirmed,
    confirm_no_default,
    header,
    print_plan,
    print_result,
)

from knowledge_agent.entity_extractors.dispatcher import extract  # noqa: E402
from knowledge_agent.entity_extractors.extractor_lifecycle import (  # noqa: E402
    install_extractor_execute,
    install_extractor_plan,
    uninstall_extractor_execute,
    uninstall_extractor_plan,
)


EXTRACTOR = "gliner"


def main() -> None:
    header(f"STEP 1: build install plan for {EXTRACTOR!r}")
    plan = install_extractor_plan(EXTRACTOR)
    print_plan(f"install_extractor_plan({EXTRACTOR!r})", plan)

    if plan.already_installed:
        print(
            f"\nAdapter {EXTRACTOR!r} already installed — skipping "
            "install step."
        )
    else:
        bail_if_not_confirmed(
            f"Proceed with pip install of {EXTRACTOR!r} "
            f"(extra {plan.pip_extras!r})?"
        )
        result = install_extractor_execute(plan)
        print_result("install_extractor_execute", result)
        if not result.install_ok:
            print(
                "\nInstall failed. Pip output above. Aborting smoke."
            )
            sys.exit(1)

    header(f"STEP 2: invoke {EXTRACTOR!r} on a sample sentence")
    print(
        "Model weights download lazily on first inference — expect "
        "a one-time ~1.1 GB pull at this step if the HF cache is "
        "empty.\n"
    )

    sample_text = (
        "Marie Curie discovered radium at the Sorbonne in 1898."
    )
    print(f"Input: {sample_text!r}")
    mentions = extract(
        text=sample_text,
        entity_types=[],  # adapter defaults: PERSON, ORG, LOC, etc.
        extractor=EXTRACTOR,
    )
    print(f"\nMentions found: {len(mentions)}")
    for m in mentions:
        print(f"  {m.raw_text!r:<25s} -> {m.entity_type}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default(f"Uninstall {EXTRACTOR!r} now?"):
        uplan = uninstall_extractor_plan(EXTRACTOR)
        print_plan(f"uninstall_extractor_plan({EXTRACTOR!r})", uplan)
        uresult = uninstall_extractor_execute(uplan)
        print_result("uninstall_extractor_execute", uresult)
        print(
            "\nNOTE: HF cache for this model stays on disk "
            "(~/.cache/huggingface). Remove via "
            "`huggingface-cli delete-cache` if disk space matters."
        )
    else:
        print(f"Keeping {EXTRACTOR!r} installed. Done.")


if __name__ == "__main__":
    main()
