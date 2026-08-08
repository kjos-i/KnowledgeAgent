"""Smoke test for the Voyage embedding provider install path.

Voyage was previously bundled in base deps. Post-2026-06-29 it's
installed via the `embed-voyage` extra, picked at first-launch
wizard time.

Same shape as the other embedder smokes — install + invoke +
optional uninstall.

REQUIRES `VOYAGE_API_KEY` in `.env.test` before running.

Run from the project root:
    python scripts/smoke_install_embedder_voyage.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Route this smoke through .env.test so its config + any download resolve from
# the isolated test box, never the real .env. Must run before any
# knowledge_agent import that resolves get_settings().
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

from knowledge_agent import embedder_factory  # noqa: E402
from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.embedder_lifecycle import (  # noqa: E402
    install_embedder_provider_execute,
    install_embedder_provider_plan,
    uninstall_embedder_provider_execute,
    uninstall_embedder_provider_plan,
)

PROVIDER = "voyage"


async def main() -> None:
    if not get_settings().voyage_api_key:
        print("VOYAGE_API_KEY is not set in .env.test. Add it before running this smoke.")
        sys.exit(1)

    plan = install_embedder_provider_plan(PROVIDER)
    print_plan(f"install_embedder_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print("\nAdapter already installed — skipping install step.")
    else:
        bail_if_not_confirmed("Proceed with pip install?")
        result = await install_embedder_provider_execute(plan)
        print_result("install_embedder_provider_execute", result)
        if not result.install_ok:
            print("\nInstall failed. Pip output above. Aborting.")
            sys.exit(1)

    header("INVOKE: embed one short text via embedder_factory")

    settings = get_settings()
    settings.embedding_provider = PROVIDER  # type: ignore[misc]
    embedder_factory.clear_cache()

    vectors = await embedder_factory.embed_texts(
        ["smoke test embedding for voyage"], input_type="document"
    )
    assert len(vectors) == 1, "expected one vector for one text"
    print(f"\nVector length: {len(vectors[0])} (expected 1024)")
    print(f"First 5 dims: {vectors[0][:5]}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the Voyage embedding adapter now?"):
        # Switch active away first so uninstall isn't blocked.
        settings.embedding_provider = "openai"  # type: ignore[misc]
        embedder_factory.clear_cache()
        uplan = uninstall_embedder_provider_plan(PROVIDER)
        print_plan("uninstall_embedder_provider_plan", uplan)
        uresult = await uninstall_embedder_provider_execute(uplan)
        print_result("uninstall_embedder_provider_execute", uresult)
    else:
        print("Keeping the adapter installed. Done.")


if __name__ == "__main__":
    asyncio.run(main())
