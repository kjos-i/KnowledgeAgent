"""Smoke test for the ontology install/import lifecycle.

Exercises the full plan/execute pattern against ONE small ontology
(ECO — Evidence & Conclusion Ontology, ~2000 classes, ~10 MB).
ECO chosen because:

  - Small enough that the download + parse complete in seconds
  - Domain tag `general` so the smoke runs regardless of corpus
    domain choice
  - Pronto-OBO format = same import path as the bulk of the 18
    shipped ontologies, so the smoke validates the most-used code
    path

Flow:

  1. `import_ontology_plan("eco")` — prints provenance (publisher,
     license, download size, cross-link surface against installed
     extractors).
  2. User confirms; `import_ontology_execute(plan)` downloads + parses
     + writes Cypher.
  3. Pause for inspection in Neo4j Browser.
  4. `delete_ontology_plan("eco")` — prints what would be wiped.
  5. User confirms; `delete_ontology_execute(plan)` surgically removes
     ECO terms + their edges.

Counterpart automation: there's no parallel unit test for the
download+parse path (that's a real network call); unit tests in
`tests/unit/kg/test_ontology_*_writes.py` cover the parsing logic
with mocked file inputs.

REQUIRES Neo4j Desktop running with the test or real instance
configured in `.env`. The smoke uses the active database; consider
running against the test instance via `python -c
"from knowledge_agent.config import load_test_env; load_test_env()"`
before launching for safety.

Run from the project root:
    python scripts/smoke_install_ontology.py
"""

from __future__ import annotations

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

from knowledge_agent.kg.ontology_lifecycle import (  # noqa: E402
    delete_ontology_execute,
    delete_ontology_plan,
    import_ontology_execute,
    import_ontology_plan,
)


ONTOLOGY = "eco"


def main() -> None:
    header(f"STEP 1: build import plan for {ONTOLOGY!r}")
    plan = import_ontology_plan(ONTOLOGY)
    print_plan(f"import_ontology_plan({ONTOLOGY!r})", plan)

    if plan.already_imported:
        print(
            f"\nOntology {ONTOLOGY!r} already imported — skipping "
            "import step. Proceeding to inspection + cleanup."
        )
    else:
        bail_if_not_confirmed(
            f"Download + import {ONTOLOGY!r} into Neo4j now?"
        )
        result = import_ontology_execute(plan)
        print_result("import_ontology_execute", result)
        if not result.import_ok:
            print(
                f"\nImport failed. Check the result above + Neo4j logs."
            )
            sys.exit(1)

    header("STEP 2: inspect in Neo4j Browser")
    print(
        f"Suggested Cypher to inspect {ONTOLOGY!r} terms:\n"
        f"  MATCH (t:ECOTerm) RETURN t LIMIT 20;\n"
        f"  MATCH (t:ECOTerm)-[r]-(o) RETURN t, r, o LIMIT 20;\n"
        f"\nNote: label is `:ECOTerm` for ECO. Other ontologies use "
        "their own label (`:HPOTerm`, `:MeSHTerm`, etc.)."
    )

    input("\nPress Enter when you're done inspecting...")

    if confirm_no_default(
        f"Delete {ONTOLOGY!r} from Neo4j now? "
        "(removes terms + their edges)"
    ):
        del_plan = delete_ontology_plan(ONTOLOGY)
        print_plan(f"delete_ontology_plan({ONTOLOGY!r})", del_plan)
        if del_plan.is_imported:
            del_result = delete_ontology_execute(del_plan)
            print_result("delete_ontology_execute", del_result)
        else:
            print(f"\n{ONTOLOGY!r} not imported — nothing to delete.")
    else:
        print(f"Keeping {ONTOLOGY!r} in Neo4j. Done.")


if __name__ == "__main__":
    main()
