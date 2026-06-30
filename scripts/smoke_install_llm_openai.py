"""Smoke test for the OpenAI LLM provider install path.

Exercises the full GUI workflow without the GUI:

  1. Build `install_llm_provider_plan("openai")` and print its summary
     (publisher, license, pip-extras name).
  2. Confirm with user, then `install_llm_provider_execute(plan)`
     which runs `pip install <pkg>[llm-openai]` against the active
     interpreter.
  3. Set `settings.llm_provider = "openai"` (in-memory) and make one
     small chat call via `llm_factory.get_llm(...).invoke("say hi")`
     to confirm dispatch + key validation work end-to-end.
  4. Pause for inspection.
  5. Offer to uninstall (default: keep).

REQUIRES `OPENAI_API_KEY` in `.env` before running.

Run from the project root:
    python scripts/smoke_install_llm_openai.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Smoke-script bootstrap: add `scripts/` to sys.path so the shared
# helper module imports cleanly when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _install_smoke_lib import (  # noqa: E402
    bail_if_not_confirmed,
    confirm_no_default,
    header,
    print_plan,
    print_result,
)

from knowledge_agent import llm_factory  # noqa: E402
from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.llm_lifecycle import (  # noqa: E402
    install_llm_provider_execute,
    install_llm_provider_plan,
    uninstall_llm_provider_execute,
    uninstall_llm_provider_plan,
)


PROVIDER = "openai"
SMOKE_MODEL = "gpt-4o-mini"  # cheapest GPT — keep smoke cost minimal


async def main() -> None:
    if not get_settings().openai_api_key:
        print(
            "OPENAI_API_KEY is not set in .env. Add it before running "
            "this smoke."
        )
        sys.exit(1)

    plan = await install_llm_provider_plan(PROVIDER)
    print_plan(f"install_llm_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print(
            "\nAdapter already installed — skipping install step. "
            "Proceeding to invocation test."
        )
    else:
        bail_if_not_confirmed("Proceed with pip install?")
        result = await install_llm_provider_execute(plan)
        print_result("install_llm_provider_execute", result)
        if not result.install_ok:
            print(
                "\nInstall failed. Pip output is above. Aborting smoke."
            )
            sys.exit(1)
        # langchain_openai needs to be importable in THIS interpreter
        # for the next step; pip-installed in subprocess + the
        # restart_required flag tells the GUI to ask the user to
        # restart. In this smoke we just import lazily below — if pip
        # said success, it should work without restart in 99% of cases.
        print("\nNOTE: restart required in the GUI; in this smoke we "
              "proceed directly via lazy import.")

    header("INVOKE: send 'say hi' via llm_factory.get_llm")

    # Force the factory to dispatch to OpenAI even if .env says
    # anthropic. The factory reads settings.llm_provider; we mutate
    # in-memory and clear the cache.
    settings = get_settings()
    settings.llm_provider = PROVIDER  # type: ignore[misc]
    llm_factory.clear_cache()

    llm = llm_factory.get_llm(SMOKE_MODEL, 0.0)
    response = llm.invoke("Say hi in five words or fewer.")
    print(f"\nResponse: {response.content!r}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the OpenAI adapter now?"):
        uplan = uninstall_llm_provider_plan(PROVIDER)
        # Switch active provider away first so the uninstall isn't
        # blocked. Mirrors the GUI's "switch active provider first"
        # message.
        settings.llm_provider = "anthropic"  # type: ignore[misc]
        llm_factory.clear_cache()
        uplan = uninstall_llm_provider_plan(PROVIDER)  # rebuild plan
        print_plan("uninstall_llm_provider_plan", uplan)
        uresult = await uninstall_llm_provider_execute(uplan)
        print_result("uninstall_llm_provider_execute", uresult)
    else:
        print("Keeping the adapter installed. Done.")


if __name__ == "__main__":
    asyncio.run(main())
