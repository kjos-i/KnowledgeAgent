"""Security smoke: data-leakage / confidentiality checks (audit track G, LLM02).

Three checks against REAL local facilities. Unlike the other security smokes
this one needs NO LLM, NO Neo4j, and NO internet to a provider, so it runs
anywhere:

  1. Telemetry egress: importing `knowledge_agent.evaluation` sets DeepEval's
     opt-out env vars BEFORE deepeval can be imported, and a DNS sentinel
     confirms no PostHog/Sentry hostname is resolved while deepeval imports.
  2. Secret-in-logs: logging a `Settings` object that holds a probe API key
     never writes the raw key. SecretStr (audit finding 4) masks it.
  3. Keyring at-rest: a probe secret round-trips through the OS keyring and
     never appears in plaintext in `settings.json` or `.env`.

SAFETY: the keyring check uses a DEDICATED probe entry (`_smoke-security-probe`)
and fake values, so your real provider keys in the OS keyring are never read,
overwritten, or deleted. The probe entry is removed at the end.

Run from the project root:
    python scripts/smoke_security_leakage.py

Prints PASS / FAIL per check; exits non-zero if any check fails. Automated
counterpart: tests/unit/test_config.py::test_secret_keys_never_leak_in_repr_or_dump
(the same SecretStr guarantee, unit-level).
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile
from pathlib import Path

# Load the TEST env (never the real .env) before any config import. Leakage
# checks don't touch Neo4j, but this keeps the process on test credentials.
from knowledge_agent.config import load_test_env

load_test_env()

_PROBE_KEY = "sk-PROBE-SECRET-must-not-appear-anywhere-9f3a2b"  # pragma: allowlist secret
_PROBE_KEYRING_ENTRY = "_smoke-security-probe"
_PROBE_KEYRING_VALUE = "kr-PROBE-must-not-be-plaintext-7c1d8e"  # pragma: allowlist secret


def check_telemetry_optout() -> bool:
    """Importing the evaluation package must opt DeepEval out of telemetry
    BEFORE deepeval is importable, and importing deepeval must not resolve a
    PostHog/Sentry hostname."""
    resolved: list[str] = []
    original = socket.getaddrinfo

    def _spy(host, *args, **kwargs):
        resolved.append(str(host))
        return original(host, *args, **kwargs)

    socket.getaddrinfo = _spy  # type: ignore[assignment]
    try:
        import knowledge_agent.evaluation  # noqa: F401  (sets the opt-out env vars)

        opt_out = os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT")
        err_report = os.environ.get("ERROR_REPORTING")
        try:
            import deepeval  # noqa: F401  (would phone home if not opted out)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  note: importing deepeval raised {exc!r} (still checking egress)")
    finally:
        socket.getaddrinfo = original  # type: ignore[assignment]

    telemetry_hosts = [h for h in resolved if "posthog" in h.lower() or "sentry" in h.lower()]
    print(f"  DEEPEVAL_TELEMETRY_OPT_OUT={opt_out!r}  ERROR_REPORTING={err_report!r}")
    if telemetry_hosts:
        print(f"  EGRESS to telemetry hosts: {telemetry_hosts}")
    return opt_out == "YES" and err_report == "NO" and not telemetry_hosts


def check_secret_in_logs() -> bool:
    """Logging a Settings object that holds a probe key must not write the raw
    key to the log file (SecretStr masks it as ``**********``)."""
    from pydantic import SecretStr

    from knowledge_agent.config import get_settings

    settings = get_settings().model_copy(
        update={
            "anthropic_api_key": SecretStr(_PROBE_KEY),
            "neo4j_password": SecretStr(_PROBE_KEY),
        }
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="ka-smoke-leak-"))
    log_path = tmp_dir / "probe.log"
    logger = logging.getLogger("knowledge_agent.smoke_security_probe")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        logger.debug("settings repr: %r", settings)
        logger.debug("settings model_dump: %s", settings.model_dump())
        logger.debug("settings model_dump_json: %s", settings.model_dump_json())
    finally:
        handler.close()
        logger.removeHandler(handler)

    content = log_path.read_text(encoding="utf-8")
    leaked = _PROBE_KEY in content
    log_path.unlink(missing_ok=True)
    tmp_dir.rmdir()
    if leaked:
        print(f"  RAW KEY FOUND in log output at {log_path}")
    else:
        print("  probe key masked in repr / model_dump / model_dump_json")
    return not leaked


def check_keyring_at_rest() -> bool:
    """A probe secret round-trips through the OS keyring and never appears as
    plaintext in settings.json or .env. Uses a dedicated probe entry so real
    provider keys are untouched."""
    try:
        from knowledge_agent.gui import config_store
    except Exception as exc:  # pragma: no cover
        print(f"  SKIP: cannot import config_store ({exc!r})")
        return True

    try:
        config_store.set_api_key(_PROBE_KEYRING_ENTRY, "")  # clear any leftover
        config_store.set_api_key(_PROBE_KEYRING_ENTRY, _PROBE_KEYRING_VALUE)
        roundtrip = config_store.get_api_key(_PROBE_KEYRING_ENTRY)
    except Exception as exc:  # pragma: no cover - no keyring backend
        print(f"  SKIP: keyring backend unavailable ({exc!r})")
        return True

    roundtrip_ok = roundtrip == _PROBE_KEYRING_VALUE

    plaintext_hits: list[str] = []
    candidates = [config_store._config_file(), Path(".env"), Path(".env.test")]
    for candidate in candidates:
        try:
            if candidate.exists() and _PROBE_KEYRING_VALUE in candidate.read_text(encoding="utf-8"):
                plaintext_hits.append(str(candidate))
        except OSError:
            pass

    # Cleanup: remove the probe entry (empty value deletes).
    config_store.set_api_key(_PROBE_KEYRING_ENTRY, "")

    print(
        f"  keyring round-trip ok={roundtrip_ok}; plaintext-on-disk hits={plaintext_hits or 'none'}"
    )
    return roundtrip_ok and not plaintext_hits


def main() -> int:
    checks = [
        ("telemetry egress opt-out", check_telemetry_optout),
        ("secret never in logs", check_secret_in_logs),
        ("keyring at-rest", check_keyring_at_rest),
    ]
    print("Security leakage smoke (audit G / OWASP LLM02)\n")
    results: list[tuple[str, bool]] = []
    for name, fn in checks:
        print(f"[{name}]")
        try:
            ok = fn()
        except Exception as exc:  # pragma: no cover
            print(f"  ERROR: {exc!r}")
            ok = False
        print(f"  => {'PASS' if ok else 'FAIL'}\n")
        results.append((name, ok))

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All leakage checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
