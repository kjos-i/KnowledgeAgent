"""Smoke test for the HuggingFace embedding provider install path (2-step).

Unique flow compared to the cloud embedder smokes — HF is the
heaviest install on the menu and has TWO install components:

  1. Libs (`sentence-transformers` + `torch`) — ~2-3 GB. Installed
     via the `embed-huggingface` pip extra.
  2. Model weights — 90 MB to 2.3 GB depending on which curated
     menu entry is picked. This smoke defaults to the smallest model
     (`all-MiniLM-L6-v2`, 90 MB / 384 dim) so it stays fast; pick
     a bigger one manually if you want to test BGE-m3 etc.

This smoke runs both steps and verifies end-to-end: embed one short
text + check the vector length matches the model's pinned dim.

Run from the project root:
    python scripts/smoke_install_embedder_huggingface.py
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
    HF_EMBEDDING_MODELS,
    download_hf_model_execute,
    download_hf_model_plan,
    install_embedder_provider_execute,
    install_embedder_provider_plan,
    uninstall_embedder_provider_execute,
    uninstall_embedder_provider_plan,
)


PROVIDER = "huggingface"
# Smallest curated model = fastest smoke. Bump to BAAI/bge-m3 etc.
# for a heavier integration test.
SMOKE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def main() -> None:
    header("STEP 1: install sentence-transformers + torch libs")
    plan = install_embedder_provider_plan(PROVIDER)
    print_plan(f"install_embedder_provider_plan({PROVIDER!r})", plan)

    if plan.already_installed:
        print("\nLibs already installed — skipping.")
    else:
        print(
            "\nNOTE: this install pulls torch — expect ~2-3 GB "
            "depending on the CUDA wheel."
        )
        bail_if_not_confirmed("Proceed with pip install?")
        result = await install_embedder_provider_execute(plan)
        print_result("install_embedder_provider_execute", result)
        if not result.install_ok:
            print("\nInstall failed. Pip output above. Aborting.")
            sys.exit(1)

    header(f"STEP 2: download model {SMOKE_MODEL}")
    print(
        f"All 4 curated HF embedding models in the menu:"
    )
    for model_id, prov in HF_EMBEDDING_MODELS.items():
        marker = " <-- this smoke uses this one" if model_id == SMOKE_MODEL else ""
        print(
            f"  {model_id:<55s} {prov.download_size_mb:>5} MB  "
            f"{prov.dimensions}-dim{marker}"
        )

    dl_plan = download_hf_model_plan(SMOKE_MODEL)
    print_plan(f"download_hf_model_plan({SMOKE_MODEL!r})", dl_plan)

    if dl_plan.libs_installed:
        bail_if_not_confirmed("Proceed with model download?")
        dl_result = download_hf_model_execute(dl_plan)
        print_result("download_hf_model_execute", dl_result)
        if not dl_result.download_ok:
            print("\nModel download failed. Aborting.")
            sys.exit(1)

    header("STEP 3: embed one short text via embedder_factory")

    settings = get_settings()
    settings.embedding_provider = PROVIDER  # type: ignore[misc]
    settings.hf_embedding_model = SMOKE_MODEL  # type: ignore[misc]
    embedder_factory.clear_cache()

    vectors = await embedder_factory.embed_texts(
        ["smoke test embedding for huggingface"], input_type="document"
    )
    assert len(vectors) == 1, "expected one vector for one text"
    expected_dim = HF_EMBEDDING_MODELS[SMOKE_MODEL].dimensions
    print(
        f"\nVector length: {len(vectors[0])} "
        f"(expected {expected_dim})"
    )
    print(f"First 5 dims: {vectors[0][:5]}")

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default("Uninstall the HuggingFace libs now?"):
        # Switch active away first so uninstall isn't blocked.
        settings.embedding_provider = "voyage"  # type: ignore[misc]
        embedder_factory.clear_cache()
        uplan = uninstall_embedder_provider_plan(PROVIDER)
        print_plan("uninstall_embedder_provider_plan", uplan)
        uresult = await uninstall_embedder_provider_execute(uplan)
        print_result("uninstall_embedder_provider_execute", uresult)
        print(
            "\nNOTE: HF cache files (model weights) remain on disk "
            "under ~/.cache/huggingface. Remove via "
            "`huggingface-cli delete-cache` if disk space matters."
        )
    else:
        print("Keeping the libs installed. Done.")


if __name__ == "__main__":
    asyncio.run(main())
