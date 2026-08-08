"""Smoke test for the Anthropic LLM provider install path.

Anthropic was previously bundled in base deps. Post-2026-06-29 it's
a provider like any other — installed via the `llm-anthropic` extra,
picked at first-launch wizard time.

Same shape as `smoke_install_llm_openai.py` — install adapter,
invoke one small call, optional uninstall.

REQUIRES `ANTHROPIC_API_KEY` in `.env.test` before running.

Run from the project root:
    python scripts/smoke_install_llm_anthropic.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Route this smoke through .env.test so its key + DB config resolve from the
# isolated test box, never the real .env. Must run before any knowledge_agent
# import that resolves get_settings().
from knowledge_agent.config import load_test_env

load_test_env()

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

PROVIDER = "anthropic"
SMOKE_MODEL = "claude-haiku-4-5-20251001"  # cheapest tier


async def main() -> None:
    if not get_settings().anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set in .env.test. Add it before running this smoke.")
        sys.exit(1)

    plan = await install_llm_provider_plan(PROVIDER)
    print_plan(f"install_llm_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print("\nAdapter already installed — skipping install step.")
    else:
        bail_if_not_confirmed("Proceed with pip install?")
        result = await install_llm_provider_execute(plan)
        print_result("install_llm_provider_execute", result)
        if not result.install_ok:
            print("\nInstall failed. Pip output above. Aborting.")
            sys.exit(1)

    header("INVOKE: send 'say hi' via llm_factory.get_llm")

    settings = get_settings()
    settings.llm_provider = PROVIDER  # type: ignore[misc]
    llm_factory.clear_cache()

    llm = llm_factory.get_llm(SMOKE_MODEL, 0.0)
    response = llm.invoke("Say hi in five words or fewer.")
    print(f"\nResponse: {response.content!r}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the Anthropic adapter now?"):
        # Switching away first lets the uninstall succeed (active
        # provider can't be uninstalled).
        settings.llm_provider = "openai"  # type: ignore[misc]
        llm_factory.clear_cache()
        uplan = uninstall_llm_provider_plan(PROVIDER)
        print_plan("uninstall_llm_provider_plan", uplan)
        uresult = await uninstall_llm_provider_execute(uplan)
        print_result("uninstall_llm_provider_execute", uresult)
    else:
        print("Keeping the adapter installed. Done.")


if __name__ == "__main__":
    asyncio.run(main())
