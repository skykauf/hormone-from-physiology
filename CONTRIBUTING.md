# Contributing

Thanks for helping build an open baseline for hormone-from-physiology research.

## Ground rules

1. **Do not commit mcPHASES data** or any other identifiable health records.
2. Prefer **leave-one-subject-out** (or other subject-wise) evaluation for model claims.
3. Keep hormone columns out of feature matrices unless the PR is an explicit ablation.
4. Document dataset access steps; never redistribute PhysioNet restricted files.

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
```

## PR checklist

- [ ] Tests added/updated
- [ ] No raw participant data in the diff
- [ ] Metrics / claims match the evaluation scheme used
- [ ] Docs updated if CLI or data requirements change
