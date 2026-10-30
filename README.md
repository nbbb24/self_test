# Self Test

This repository is a personal collection of reproducible validation scripts,
small experiments, and debugging utilities for inference frameworks and their
dependencies.

The tests here are intentionally independent from upstream CI. They are meant
for manual execution on dedicated hardware, especially when a scenario needs
large models, multiple accelerators, special networking, or external storage
services.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [`sglang/`](sglang/) | SGLang deployment and feature validation scripts. |
| [`vllm/`](vllm/) | vLLM validation scripts and experiments. |
| [`memfabric/h2d_d2h/`](memfabric/h2d_d2h/) | Python allocation/copy interfaces and local H2D/D2H validation for MemFabric. |

Each test directory should contain:

- A README describing the purpose, prerequisites, commands, and pass criteria.
- Scripts that accept environment-variable overrides instead of embedding
  machine-specific paths or addresses.
- Explicit output artifacts so a result can be reviewed after the run.
- A non-zero exit code when validation fails.

## Conventions

- Never commit model weights, credentials, host inventories, or generated logs.
- Prefer deterministic requests and correctness checks over launch-only smoke
  tests.
- Record the relevant framework revision and hardware generation in test notes.
- Keep framework-specific content under its own top-level directory.
