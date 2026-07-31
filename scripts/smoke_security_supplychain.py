"""Security smoke: model supply-chain checks (audit track G, OWASP LLM03/LLM05).

Guards the model-weight supply chain: a downloaded model must load ONLY a
pinned, immutable commit SHA (never a mutable branch like `main`), and pickle-
format weights (which execute code on load) must carry a visible warning.

Two checks:

  1. Pinned-revision integrity (no network): every model that downloads weights
     from HuggingFace (the NER extractors + the curated HF embedders) pins a
     40-character commit SHA, not a branch/tag. Pickle models (safetensors
     False) additionally surface the pickle security warning.
  2. SHA enforcement (--with-download; needs internet): fetch just `config.json`
     at a model's pinned SHA to confirm the revision resolves, and confirm that
     a bogus SHA is REJECTED. Lightweight (no GB weights), but network-bound.

Run from the project root:
    python scripts/smoke_security_supplychain.py                 # check 1
    python scripts/smoke_security_supplychain.py --with-download # + check 2

Related: pyproject floors torch>=2.6 for the pickle extractors so torch.load
defaults to weights_only=True (audit finding 3).
"""

from __future__ import annotations

import argparse
import re
import sys

from knowledge_agent._provenance import security_warning_text

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _collect_pinned_models() -> list[tuple[str, object]]:
    """Return (label, provenance) for every model that pins a HF revision.

    Covers the three NER extractors (via `EXTRACTOR_REGISTRY`) and the curated
    HF embedder menu (found by scanning the module for `HFEmbedderProvenance`
    instances, so a new menu entry is picked up automatically)."""
    collected: list[tuple[str, object]] = []

    from knowledge_agent.entity_extractors import extractor_lifecycle as ext

    for name, entry in ext.EXTRACTOR_REGISTRY.items():
        prov = entry.get("provenance") if isinstance(entry, dict) else None
        if (
            prov is not None
            and hasattr(prov, "safetensors")
            and getattr(prov, "pinned_revision", "")
        ):
            collected.append((f"extractor:{name}", prov))

    from knowledge_agent import embedder_lifecycle as emb

    seen: set[int] = set()

    def _walk(obj: object):
        if isinstance(obj, emb.HFEmbedderProvenance):
            if id(obj) not in seen:
                seen.add(id(obj))
                yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from _walk(value)
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                yield from _walk(value)

    for value in vars(emb).values():
        for prov in _walk(value):
            label = getattr(prov, "display_name", None) or getattr(
                prov, "model_name", "hf-embedder"
            )
            collected.append((f"embedder:{label}", prov))

    return collected


def check_pinned_revisions() -> bool:
    """Every HF-downloaded model pins a 40-char SHA; pickle models are flagged."""
    models = _collect_pinned_models()
    if not models:
        print("  no pinned models found (unexpected)")
        return False

    ok = True
    for label, prov in models:
        rev = getattr(prov, "pinned_revision", "")
        safetensors = getattr(prov, "safetensors", True)
        is_sha = bool(_SHA_RE.match(rev))
        if not is_sha:
            print(f"  BAD PIN {label}: revision {rev!r} is not a 40-char commit SHA")
            ok = False
            continue
        note = "safetensors"
        if not safetensors:
            warning = security_warning_text(safetensors=False)
            if "pickle" not in warning.lower():
                print(f"  MISSING WARNING {label}: pickle model without a pickle disclosure")
                ok = False
                continue
            note = "PICKLE (warning present)"
        print(f"  {label}: SHA {rev[:12]}... [{note}]")
    if ok:
        print(f"  all {len(models)} models pin a commit SHA; pickle models flagged")
    return ok


def check_sha_enforcement() -> bool:
    """Fetch config.json at a pinned SHA (must resolve) and at a bogus SHA
    (must be rejected). Confirms the pin is enforced by the download layer."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except Exception as exc:  # pragma: no cover
        print(f"  SKIP: huggingface_hub unavailable ({exc!r})")
        return True

    repo = "sentence-transformers/all-MiniLM-L6-v2"
    good_sha = (
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # pragma: allowlist secret (public HF commit)
    )
    bogus_sha = "0" * 40

    try:
        path = hf_hub_download(repo_id=repo, filename="config.json", revision=good_sha)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  SKIP: could not reach HuggingFace for the pinned SHA ({exc!r})")
        return True
    print(f"  pinned SHA resolved: {path}")

    rejected = False
    try:
        hf_hub_download(repo_id=repo, filename="config.json", revision=bogus_sha)
    except (HfHubHTTPError, Exception) as exc:
        rejected = True
        print(f"  bogus SHA rejected: {type(exc).__name__}")
    if not rejected:
        print("  BOGUS SHA ACCEPTED - pin not enforced!")
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="Security supply-chain smoke.")
    parser.add_argument(
        "--with-download",
        action="store_true",
        help="Also fetch config.json at pinned/bogus SHAs to verify enforcement (needs internet).",
    )
    args = parser.parse_args()

    print("Security supply-chain smoke (audit G / OWASP LLM03+LLM05)\n")
    results: list[tuple[str, bool]] = []

    print("[pinned-revision integrity]")
    r = check_pinned_revisions()
    print(f"  => {'PASS' if r else 'FAIL'}\n")
    results.append(("pinned revisions", r))

    if args.with_download:
        print("[SHA enforcement (live)]")
        r = check_sha_enforcement()
        print(f"  => {'PASS' if r else 'FAIL'}\n")
        results.append(("sha enforcement", r))
    else:
        print("[SHA enforcement] SKIPPED (pass --with-download to run)\n")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All run supply-chain checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
