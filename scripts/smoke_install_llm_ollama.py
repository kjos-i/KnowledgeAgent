"""Smoke test for the Ollama LLM provider install path (3-step).

Unique flow compared to the cloud LLM smokes — Ollama has THREE
install components and only one is pip-installable:

  1. Daemon binary (Go, ~150-300 MB) — NOT pip-installable. Detected
     via `_ollama_daemon_is_reachable()`. If missing, the smoke
     surfaces the manual install URL and aborts.
  2. `langchain-ollama` Python adapter (~1-2 MB) — installed via the
     `llm-ollama` pip extra.
  3. Model weights (2-40 GB per model) — pulled via `ollama pull`.
     The smoke does NOT auto-pull (too big); it lists the curated
     menu so the user can pull manually if they want to test
     end-to-end.

REQUIRES the Ollama daemon installed manually from
https://ollama.com/download BEFORE running this smoke.

Run from the project root:
    python scripts/smoke_install_llm_ollama.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _install_smoke_lib import (  # noqa: E402
    bail_if_not_confirmed,
    confirm,
    confirm_no_default,
    header,
    print_plan,
    print_result,
)

from knowledge_agent import llm_factory  # noqa: E402
from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.llm_lifecycle import (  # noqa: E402
    OLLAMA_MODELS,
    _ollama_daemon_is_reachable,
    install_llm_provider_execute,
    install_llm_provider_plan,
    pull_ollama_model_execute,
    pull_ollama_model_plan,
    uninstall_llm_provider_execute,
    uninstall_llm_provider_plan,
)


PROVIDER = "ollama"


def main() -> None:
    header("STEP 1: detect Ollama daemon")
    daemon_up = _ollama_daemon_is_reachable()
    print(f"Daemon reachable: {daemon_up}")
    if not daemon_up:
        print(
            "\nOllama daemon is not on PATH and not responding at the "
            "configured base URL. Install from "
            "https://ollama.com/download, start it, then re-run."
        )
        sys.exit(1)

    header("STEP 2: install langchain-ollama adapter")
    plan = install_llm_provider_plan(PROVIDER)
    print_plan(f"install_llm_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print(
            "\nAdapter already installed — skipping install step."
        )
    else:
        bail_if_not_confirmed("Proceed with pip install?")
        result = install_llm_provider_execute(plan)
        print_result("install_llm_provider_execute", result)
        if not result.install_ok:
            print("\nInstall failed. Pip output above. Aborting.")
            sys.exit(1)

    header("STEP 3: curated Ollama model menu (informational)")
    print(
        "Available curated models (run `ollama pull <id>` manually "
        "if you need to download one before invoking):"
    )
    for model_id, prov in OLLAMA_MODELS.items():
        print(
            f"  {model_id:<20s} {prov.download_size_gb:5.1f} GB  "
            f"({prov.hardware_tier}, min {prov.min_ram_gb} GB RAM)"
        )

    header("STEP 4: pick a model + try invoking")
    print(
        "Enter the Ollama model tag to use for the invocation test "
        "(must already be pulled via `ollama pull`). Leave blank to "
        "skip the invocation step."
    )
    chosen_model = input("Model tag (e.g. llama3.2:3b): ").strip()
    if chosen_model:
        # Try the smoke pull-plan for this model so the user sees
        # what `ollama pull` would do — but only execute pull if
        # they confirm (smoke shouldn't auto-pull GB-scale files).
        if chosen_model in OLLAMA_MODELS:
            pull_plan = pull_ollama_model_plan(chosen_model)
            print_plan(
                f"pull_ollama_model_plan({chosen_model!r})", pull_plan,
            )
            if confirm(
                f"Run `ollama pull {chosen_model}` now? "
                "(skip if already pulled)"
            ):
                pull_result = pull_ollama_model_execute(pull_plan)
                print_result("pull_ollama_model_execute", pull_result)

        settings = get_settings()
        settings.llm_provider = PROVIDER  # type: ignore[misc]
        llm_factory.clear_cache()

        llm = llm_factory.get_llm(chosen_model, 0.0)
        print(f"\nInvoking {chosen_model}: 'say hi'")
        response = llm.invoke("Say hi in five words or fewer.")
        print(f"\nResponse: {response.content!r}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the Ollama adapter now?"):
        settings = get_settings()
        settings.llm_provider = "anthropic"  # type: ignore[misc]
        llm_factory.clear_cache()
        uplan = uninstall_llm_provider_plan(PROVIDER)
        print_plan("uninstall_llm_provider_plan", uplan)
        uresult = uninstall_llm_provider_execute(uplan)
        print_result("uninstall_llm_provider_execute", uresult)
    else:
        print(
            "Keeping the adapter installed. "
            "(Pulled models stay on disk; remove via "
            "`ollama rm <model>` if needed.)"
        )


if __name__ == "__main__":
    main()
