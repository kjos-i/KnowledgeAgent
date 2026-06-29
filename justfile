# ResearchLiteratureAgent task runner.
#
# Install `just` once (Windows): winget install --id Casey.Just
# Run `just` (no args) to list available recipes.

# Show all recipes.
default:
    @just --list

# Install the package + dev dependencies in editable mode.
install:
    pip install -e ".[dev]"

# Run the test suite.
test:
    pytest tests/

# Lint the source.
lint:
    ruff check src tests

# Auto-format the source.
format:
    ruff format src tests
