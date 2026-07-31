# Verification & Audit Checklist

Run groups top to bottom for a full verify pass (cheapest and fastest first). All commands run from `KnowledgeAgent/` with the venv Python.

## Automated gates (every change)
- **Ruff lint** (`ruff check .`): bugs + style (F, E/W, I, B, UP, SIM, TCH, RUF); `--fix` auto-fixes. ("lint" and "ruff" are the same check.)
- **Ruff format** (`ruff format .`): the only formatter (double quotes, 100 cols). There is no "pytest format".
- **Pre-commit hooks** (`pre-commit run --all-files`): ruff lint + format plus hygiene (trailing whitespace, EOF, merge-conflict, yaml/toml parse, over-1MB block, private-key + detect-secrets scan).
- **Unit tests** (`pytest`): fast, no real services (I/O mocked, env isolated to `.env.test`). Includes an async-consistency AST guard, hypothesis property tests, and fails on any un-awaited coroutine. The everyday gate.
- **Security scans** (`bandit -r src/knowledge_agent -ll` + detect-secrets via pre-commit): static risky-pattern scan at medium+ severity, plus the committed-secret scan. Install the tools with `pip install -e ".[security]"`. The low-severity bandit findings are known-safe idioms (subprocess arg-lists); the one B608 in `evaluation/ledger.py` is `# nosec`'d as a verified false positive. See audit **F**.

## Deeper tests (on demand / before a release)
- **Coverage** (`pytest --cov=knowledge_agent --cov-report=term-missing`): untested lines/paths.
- **Integration** (`pytest -m integration`): real Neo4j test instance + LanceDB (plus keys); the only coverage for `kg/openalex_writes.py` and `kg/chunk_writes.py`.
- **End-to-end** (`pytest -m e2e`): launches the Flet app.
- **Fast subset** (`pytest -m "not slow"`): skip the over-1s tests.
- **Security regression** (`pytest -m security`): the `security`-marked set (Cypher-guard, LanceDB filter escaping, secret redaction, prompt fence, KG provenance). A cross-cutting marker; the unit ones also run in the everyday `pytest`.
- **Dependency CVEs** (`pip-audit`): known-vulnerable packages in the installed dependency tree. Advisory; run before a release. See audit **F**.

## Smoke scripts (manual, hit REAL services): `python scripts/smoke_*.py`
- **Ingest / pipeline**: smoke_parse, smoke_pipeline, smoke_lancedb, smoke_metadata.
- **KG layers L1 to L10**: smoke_kg_l1_l5, l6_entities, l7_xrefs_l10, l8_triples, l9_cross_doc.
- **Agent / eval / bulk / export**: smoke_agent, smoke_eval, smoke_bulk_ops, smoke_artifacts.
- **Multimodal / extractors**: smoke_multimodal, smoke_multi_extractor.
- **Provider install lifecycle**: smoke_install_llm_*, smoke_install_embedder_*, smoke_install_extractor, smoke_install_ontology.
- **Security**: `smoke_security_leakage` (telemetry egress, secret-in-logs, keyring at-rest; local, no LLM/Neo4j). `smoke_security_injection` (Cypher-guard battery, filter escaping, live Neo4j read-only belt; `--with-llm` adds agent prompt-injection). `smoke_security_supplychain` (pinned-SHA integrity + pickle flags; `--with-download` adds live SHA-enforcement).

## Audits (manual, periodic): each returns a DIFFERENT kind of information
- **A. Adversarial correctness audit** (multi-agent): confirmed logic / correctness bugs and intent-vs-code mismatches, each verified by 2 or more independent skeptics before it is reported. Method: per-subsystem finders + 2 skeptics per finding + a synthesizer. Answers **"what is broken?"**
- **B. Codebase-health / structural audit**: a descriptive survey of file-size distribution, module organization and coupling smells, bloat / hot spots, missing standard infrastructure (health check, schema migrations, backup / restore, retry / backoff, CLI, ARCHITECTURE.md, central HTTP client), professionalism and scale-readiness, and missing features. Answers **"is it well-organized, complete, professional, scalable, and what is structurally missing?"** It does NOT find logic bugs. (The early "Backend Code Audit" was this type.)
- **C. Icon-verify loop**: info-icon / help-text accuracy against the real code (a clean-room agent refutes each claim). Answers **"is the help text true?"**
- **D. Test-coverage / quality audit** (optional): what is untested, and whether tests assert the right behavior rather than just passing. Uses coverage plus reading. Answers **"do we actually test this?"**
- **E. Type-check sweep** (pyright; not wired today): static type errors. Answers **"do the types line up?"**
- **F. Security scan (automated):** `bandit` (static risky-pattern) + `pip-audit` (dependency CVEs) + `detect-secrets` (committed secrets), installed via the `[security]` extra. Answers **"any known-vulnerable dependencies or risky code patterns?"** No `semgrep`: redundant with `bandit` + audit G at this size, and it has no native-Windows support; reconsider only alongside a CI-Linux pipeline.
- **F/G shipped a first pass 2026-07-31** (`5f28684`): no reachable exploit; 6 defense-in-depth fixes (SHOW block, LanceDB filter escaping, torch>=2.6 pin, SecretStr keys, prompt fence, KG provenance), each locked by a `security`-marked test.
- **G. Adversarial security review** (manual / agentic): per-surface finders (Cypher-injection, prompt-injection, subprocess, pickle / supply-chain, secrets / telemetry, untrusted parsing, path-trust, LanceDB-filter, RAG poisoning), each verified by independent skeptics, mapped to the OWASP 2025 LLM Top 10. Answers **"can untrusted input execute code, inject queries, or exfiltrate secrets?"** Excessive Agency + Unbounded Consumption are handled by architecture (the agent binds no tools; the graph is acyclic), not a finder. Platform-specific paths (keyring backends, file-open) get cross-OS validation via a CI matrix at packaging time; the rest is OS-agnostic (mock the platform).
