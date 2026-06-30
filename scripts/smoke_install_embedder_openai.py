"""Smoke test for the OpenAI embedding provider install path.

Same shape as `smoke_install_llm_openai.py` but for the embedding
side. Installs the `embed-openai` extra, then calls
`await embedder_factory.embed_texts(["..."])` to confirm dispatch +
key validation work end-to-end.

NOTE on dimensions: OpenAI's `text-embedding-3-small` produces
1536-dim vectors by default; your current LanceDB chunks-table pins
1024 (from Voyage). Switching embedding provider with a different
dim is a DESTRUCTIVE swap — see
`embedder_lifecycle.switch_embedder_plan`. This smoke does NOT
exercise the switch (only install + invoke); it mutates
`settings.embedding_provider` in-memory just enough to dispatch one
test call and then resets.

REQUIRES `OPENAI_API_KEY` in `.env` before running.

Run from the project root:
    python scripts/smoke_install_embedder_openai.py
"""

from __future__ import annotations

import asyncio
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

from knowledge_agent import embedder_factory  # noqa: E402
from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.embedder_lifecycle import (  # noqa: E402
    install_embedder_provider_execute,
    install_embedder_provider_plan,
    uninstall_embedder_provider_execute,
    uninstall_embedder_provider_plan,
)


PROVIDER = "openai"


async def main() -> None:
    if not get_settings().openai_api_key:
        print(
            "OPENAI_API_KEY is not set in .env. Add it before running "
            "this smoke."
        )
        sys.exit(1)

    plan = install_embedder_provider_plan(PROVIDER)
    print_plan(f"install_embedder_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print(
            "\nAdapter already installed — skipping install step."
        )
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
        ["smoke test embedding for openai"], input_type="document"
    )
    assert len(vectors) == 1, "expected one vector for one text"
    print(f"\nVector length: {len(vectors[0])} (expected 1536)")
    print(f"First 5 dims: {vectors[0][:5]}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the OpenAI embedding adapter now?"):
        settings.embedding_provider = "voyage"  # type: ignore[misc]
        embedder_factory.clear_cache()
        uplan = uninstall_embedder_provider_plan(PROVIDER)
        print_plan("uninstall_embedder_provider_plan", uplan)
        uresult = await uninstall_embedder_provider_execute(uplan)
        print_result("uninstall_embedder_provider_execute", uresult)
    else:
        print("Keeping the adapter installed. Done.")


if __name__ == "__main__":
    asyncio.run(main())
