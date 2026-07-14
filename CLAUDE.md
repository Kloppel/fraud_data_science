# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a compact workspace for the IEEE-CIS Fraud Detection Kaggle competition. It currently contains only the raw competition data and a download script — there is no modeling/analysis code yet. When adding analysis code, keep it as clearly named Python modules or notebooks at the repository root until a larger package structure is actually needed.

## Commands

Use the existing `fraud` conda environment:

```bash
conda activate fraud
```

Download or refresh competition data (via `kagglehub`):

```bash
python download_data.py
```

Alternative, using the Kaggle CLI directly:

```bash
kaggle competitions download -c ieee-fraud-detection -p data
```

No test framework is currently configured. For new data-processing code, add focused tests under `tests/` using `pytest` (files named `test_*.py`), preferring small fixture data over full CSV inputs, and covering at minimum: joins on `TransactionID`, feature transformations, and submission formatting. Run with:

```bash
pytest
```

There is no build step for this repository.

## Data architecture

Files live in `data/` and are not intended to be hand-edited — regenerate via `download_data.py`:

- `train_transaction.csv` / `test_transaction.csv` — one row per `TransactionID`, ~393 columns. Includes `TransactionDT` (a **timedelta** from a fixed reference point, not a real timestamp — do not treat it as a calendar date without conversion), `TransactionAmt`, and the `isFraud` target (train only).
- `train_identity.csv` / `test_identity.csv` — identity/device metadata keyed by `TransactionID`. Not every transaction has a matching identity row, so joins must be left joins from the transaction table.
- `sample_submission.csv` — expected submission format (`TransactionID`, `isFraud`).

**Column-naming gotcha:** identity columns are named with underscores in the train file (`id_01`...`id_38`) but with hyphens in the test file (`id-01`...`id-38`). Any code that concatenates or maps columns across train/test identity data must normalize this naming difference first.

Known categorical columns (treat as categorical, not numeric, in any feature pipeline):
- Transaction: `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`
- Identity: `DeviceType`, `DeviceInfo`, `id_12`–`id_38`

See `kaggle_overview.md` for the full dataset description.

## Conventions

- Python style: PEP 8, 4-space indentation, descriptive `snake_case` names for functions, variables, scripts, and generated output files.
- Prefer small, explicit scripts over hidden notebook state when producing repeatable results.
- Keep Kaggle credentials (`kaggle.json`, tokens) outside the repository — use `~/.kaggle` or environment variables. If `download_data.py` fails with `403 Forbidden`, confirm the authenticated Kaggle account has accepted the competition rules.
- Avoid committing generated caches, temporary outputs, or large derived artifacts unless intentionally part of the project.
