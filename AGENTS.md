# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact workspace for the IEEE-CIS fraud detection Kaggle data.

- `download_data.py` downloads the `ieee-fraud-detection` competition files with `kagglehub`.
- `kaggle_overview.md` summarizes the dataset, file meanings, and important categorical columns.
- `data/` contains the downloaded competition CSV files: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`, and `sample_submission.csv`.

Keep reusable analysis code in clearly named Python modules or notebooks at the repository root until a larger package structure is needed. Avoid committing generated caches, temporary outputs, or large derived artifacts unless they are intentionally part of the project.

## Build, Test, and Development Commands

Use the existing `fraud` conda environment:

```bash
conda activate fraud
```

Download or refresh competition data:

```bash
python download_data.py
```

If using the Kaggle CLI directly:

```bash
kaggle competitions download -c ieee-fraud-detection -p data
```

Inspect available files:

```bash
ls -lh data/
```

There is no build step for the current repository.

## Coding Style & Naming Conventions

Write Python using PEP 8 conventions with 4-space indentation. Use descriptive `snake_case` names for functions, variables, scripts, and generated output files. Prefer small, explicit scripts over hidden notebook state when producing repeatable results. Keep configuration such as Kaggle credentials outside the repository; use `~/.kaggle` or environment variables.

## Testing Guidelines

No test framework is currently configured. For new data-processing code, add focused tests under `tests/` using `pytest`, with test files named `test_*.py`. Prefer small fixture data over full CSV inputs. At minimum, test joins on `TransactionID`, feature transformations, and submission formatting.

Run tests with:

```bash
pytest
```

## Commit & Pull Request Guidelines

No usable local Git history was available when this guide was created, so no project-specific commit convention could be inferred. Use short, imperative commit messages such as `Add transaction feature checks` or `Document Kaggle data download`.

Pull requests should describe the purpose, list data or command changes, mention any new dependencies, and include validation steps such as `python download_data.py` or `pytest`. Link related issues when applicable.

## Security & Configuration Tips

Do not commit Kaggle API tokens, `kaggle.json`, access tokens, or local credential files. If downloads fail with `403 Forbidden`, confirm that the authenticated Kaggle account has accepted the competition rules.
