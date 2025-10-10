# Repository Guidelines

## Project Structure & Module Organization
The modeling workspace centers on Python pipelines stored in the repository root (for example `extract.py`, `merge_categories.py`, `semantic_clustering.py`). Shared utilities live in `utils.py`, while `models.py` supplies the Pydantic schemas used across stages. Streamlit UI components reside under `web/` with `entrypoint.py`, `pages/`, and `shared_utils.py` powering the dashboard flow. Generated artifacts and intermediate batches land in `results/`; treat them as disposable outputs. Environment defaults resolve in `config.py`, which reads `.env` values such as `ROOT_DIR`.

## Build, Test, and Development Commands
- `uv sync` installs locked dependencies declared in `pyproject.toml` and `uv.lock`.
- `uv run streamlit run web/entrypoint.py` starts the exploratory dashboard locally.
- `make extract` executes the Inspect evaluation defined in `extract.py@extract`; verify the refreshed keywords in `results/keywords`.
- `make categorise` and `make merge-<n>` drive downstream category aggregation; inspect run logs under `logs/`.
- `make clean-extract` deletes stale keyword outputs—run before reruns to avoid mixing batches.

## Coding Style & Naming Conventions
Follow PEP 8 defaults: 4-space indentation, snake_case for modules and functions, PascalCase for classes and Pydantic models. Prefer type hints and Pydantic validation for structured outputs. Keep CLI entrypoints thin—orchestrate work in functions inside `process.py` or `semantic_clustering.py`, and reuse helpers from `utils.py`. Rename new scripts to mirror workflow stages (`extract`, `categorise`, `merge`, `visualisation`) so Make targets remain discoverable.

## Testing Guidelines
Formal tests are minimal; add new coverage with `pytest` beneath a `tests/` directory and run via `uv run pytest`. For pipeline steps, capture deterministic fixtures in `results/<stage>-batches/` and validate them with Inspect evals (`uv run inspect eval ...`). Document expected metrics in `metric.py`, and update notebooks such as `demo.ipynb` only when outputs change.

## Commit & Pull Request Guidelines
Use imperative, scope-rich summaries (for example `refactor: tighten keyword loaders`) rather than the historic `update`. Reference relevant scripts or pages in the subject line and add concise context paragraphs when logic changes. PRs should include a summary of pipeline impact, links to related issues or Slurm job IDs, screenshots or GIFs for Streamlit UI tweaks, and confirmation that `uv sync` plus key `make` targets succeeded.

## Configuration & Data Handling
Store secrets in `.env`; never commit real keys. `config.py` expects `ROOT_DIR` to point at the project root and an OpenAI-compatible endpoint defined by `OPENAI_BASE_URL`. Generated JSON/JSONL files in `results/` may contain sensitive grant data—scrub them before sharing outside the team.
