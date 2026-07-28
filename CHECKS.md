# Verification & Audit Checklist

Run groups top to bottom for a full verify pass (cheapest and fastest first). All commands run from `KnowledgeAgent/` with the venv Python.

## Automated gates (every change)
- **Ruff lint** (`ruff check .`): bugs + style (F, E/W, I, B, UP, SIM, TCH, RUF); `--fix` auto-fixes. ("lint" and "ruff" are the same check.)
- **Ruff format** (`ruff format .`): the only formatter (double quotes, 100 cols). There is no "pytest format".
- **Pre-commit hooks** (`pre-commit run --all-files`): ruff lint + format plus hygiene (trailing whitespace, EOF, merge-conflict, yaml/toml parse, over-1MB block, private-key + detect-secrets scan).
- **Unit tests** (`pytest`): fast, no real services (I/O mocked, env isolated to `.env.test`). Includes an async-consistency AST guard, hypothesis property tests, and fails on any un-awaited coroutine. The everyday gate.

## Deeper tests (on demand / before a release)
- **Coverage** (`pytest --cov=knowledge_agent --cov-report=term-missing`): untested lines/paths.
- **Integration** (`pytest -m integration`): real Neo4j test instance + LanceDB (plus keys); the only coverage for `kg/openalex_writes.py` and `kg/chunk_writes.py`.
- **End-to-end** (`pytest -m e2e`): launches the Flet app.
- **Fast subset** (`pytest -m "not slow"`): skip the over-1s tests.

## Smoke scripts (manual, hit REAL services): `python scripts/smoke_*.py`
- **Ingest / pipeline**: smoke_parse, smoke_pipeline, smoke_lancedb, smoke_metadata.
- **KG layers L1 to L10**: smoke_kg_l1_l5, l6_entities, l7_xrefs_l10, l8_triples, l9_cross_doc.
- **Agent / eval / bulk / export**: smoke_agent, smoke_eval, smoke_bulk_ops, smoke_artifacts.
- **Multimodal / extractors**: smoke_multimodal, smoke_multi_extractor.
- **Provider install lifecycle**: smoke_install_llm_*, smoke_install_embedder_*, smoke_install_extractor, smoke_install_ontology.

## Audits (manual, periodic): each returns a DIFFERENT kind of information
- **A. Adversarial correctness audit** (multi-agent): confirmed logic / correctness bugs and intent-vs-code mismatches, each verified by 2 or more independent skeptics before it is reported. Method: per-subsystem finders + 2 skeptics per finding + a synthesizer. Answers **"what is broken?"**
- **B. Codebase-health / structural audit**: a descriptive survey of file-size distribution, module organization and coupling smells, bloat / hot spots, missing standard infrastructure (health check, schema migrations, backup / restore, retry / backoff, CLI, ARCHITECTURE.md, central HTTP client), professionalism and scale-readiness, and missing features. Answers **"is it well-organized, complete, professional, scalable, and what is structurally missing?"** It does NOT find logic bugs. (The early "Backend Code Audit" was this type.)
- **C. Icon-verify loop**: info-icon / help-text accuracy against the real code (a clean-room agent refutes each claim). Answers **"is the help text true?"**
- **D. Test-coverage / quality audit** (optional): what is untested, and whether tests assert the right behavior rather than just passing. Uses coverage plus reading. Answers **"do we actually test this?"**
- **E. Type-check sweep** (pyright; not wired today): static type errors. Answers **"do the types line up?"**
