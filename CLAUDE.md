# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A documentation and learning resource about modern AI architectures and multi-agent communication patterns. Content is organized into:
- `docs/` — Markdown reference guides for 19 AI model types and 7 communication protocols
- `examples/` — Working Python implementations (multi-agent use cases and communication patterns)
- `notebooks/` — Jupyter notebooks for interactive learning
- `tests/` — Pytest suite that validates documentation structure and example syntax

There is no installable package. The test coverage target (`--cov=examples`) applies to the `examples/` directory.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt    # includes full.txt + test/lint tools
cp .env.example .env                   # add OPENAI_API_KEY, ANTHROPIC_API_KEY
```

For minimal setup without AI libraries:
```bash
pip install -r requirements/minimal.txt
```

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_documentation.py -v

# Run a single test
pytest tests/test_examples.py::test_customer_service_agents_structure -v

# Format code
black examples/ tests/
isort examples/ tests/

# Lint checks (as run in CI)
black --check --diff examples/ tests/
isort --check-only --diff examples/ tests/
flake8 examples/ tests/ --max-line-length=100 --extend-ignore=E203,W503

# Type checking
mypy examples/ --ignore-missing-imports

# Security scan
bandit -r examples/ -ll

# Pre-commit (runs all hooks)
pre-commit run --all-files
```

## Code Style

- **Line length**: 100 (black + isort + flake8 all configured to 100)
- **Import order**: `isort` with `--profile=black`
- **Python target**: 3.8+ (`pyupgrade --py38-plus` is enforced via pre-commit)
- **Type hints**: optional but encouraged; mypy runs with `ignore_missing_imports=true` and `disallow_untyped_defs=false`

## Architecture of Examples

All use-case examples under `examples/use-cases/` follow the same structural pattern:

1. **`Enum` classes** define agent roles (`AgentType`) and message kinds (`MessageType`)
2. **`@dataclass` structs** define the message envelope (`Message`) with sender, recipient, type, content, context, and timestamp
3. **Agent classes** each hold their `AgentType`, a reference to a shared message queue/bus, and an `async process_message()` method
4. **An orchestrator class** owns all agents, dispatches messages between them, and drives the `asyncio` event loop
5. Examples import `openai` and call `client.chat.completions.create()` — no Anthropic SDK is used in the examples themselves

The `examples/communication/` directory demonstrates lower-level patterns: `mcp_implementation.py` provides a standalone `MCPServer`/`MCPClient` with `Resource` and `Tool` dataclasses; `multi_model_pipeline.py` shows chaining multiple model calls sequentially.

## Test Suite Conventions

- `test_documentation.py` — asserts that specific markdown files exist and have content; no mocking
- `test_examples.py` — compiles each example file with `compile()` (syntax check only) and spot-checks that key classes exist via `importlib`; marked `@pytest.mark.asyncio` where needed
- `test_imports.py` — verifies AI library imports succeed or skips gracefully if not installed; all tests pass even without optional deps

`pytest-asyncio` is required; async tests use `@pytest.mark.asyncio`. The `conftest.py` only registers markers.

## Documentation Structure Conventions

Each model doc in `docs/models/` follows the sections: Overview → Key Characteristics → Core Capabilities → Examples → Popular Models → Enterprise Applications → Best Practices → Code Resources.

Each protocol doc in `docs/protocols/` documents the communication pattern with implementation examples in both Python and TypeScript where applicable.

When adding a new model or protocol doc, update `README.md`'s table of contents and the relevant section list to keep `test_documentation.py` passing (it checks for specific filenames).

## Commit Message Convention

Prefix commits with a type tag:
- `Add:` new features or content
- `Fix:` bug fixes
- `Update:` changes to existing content
- `Docs:` documentation-only changes
- `Style:` formatting only
- `Refactor:` code restructuring
- `Test:` test additions or changes
