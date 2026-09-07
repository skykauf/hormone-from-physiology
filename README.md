# Hormone from Physiology

Open research stack for inferring **menstrual hormonal state from wearable physiology**, trained and evaluated on the public **[mcPHASES](https://physionet.org/content/mcphases/1.0.0/)** dataset.

This is **not** a clone of Clair Health (or any commercial product). It is a transparent baseline for the research problem they describe: map multi-day wearable signals → hormone-verified cycle phase / LH surge / PdG rise.

## What this repo does

| Capability | Status |
|---|---|
| Load mcPHASES daily hormone + Fitbit panel | ✅ |
| Cycle segmentation + complete-cycle filtering | ✅ |
| Wearable features (RHR, nightly temp, rolling / baseline deltas) | ✅ |
| Hormone labels (phase, LH surge heuristic, PdG rise) | ✅ |
| Leave-one-subject-out phase classifier | ✅ |
| Leave-one-subject-out LH-surge detector | ✅ |
| Multimodal (HRV, EDA, bioimpedance, acoustic, …) | ❌ (not in mcPHASES / commodity Fitbit export) |
| Quantitative E2/P4 in pg/mL | ❌ (needs denser clinical pairing) |

## Dataset (required, not bundled)

mcPHASES is hosted on PhysioNet under a **Restricted Health Data Use Agreement**. We do **not** redistribute it.

1. Create a PhysioNet account and request access:  
   https://physionet.org/content/mcphases/1.0.0/
2. Download the CSVs and place at least these under `data/raw/`:
   - `hormones_and_selfreport.csv`
   - `resting_heart_rate.csv`
   - `computed_temperature.csv`
3. Or set `MCPHASES_DATA_DIR` to the folder containing those files.

See [docs/data.md](docs/data.md) for schema notes and citation.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# after placing mcPHASES CSVs in data/raw/
hfp-train-phase --out artifacts/phase_model.joblib
hfp-eval --out artifacts/lh_surge_model.joblib
```

Run unit tests (synthetic fixtures, no PhysioNet download needed):

```bash
pytest
```

## Project layout

```
src/hfp/
  data/       # loaders, cycles, features
  labels/     # phase / LH / PdG ground-truth helpers
  models/     # LOSO baselines (HistGradientBoosting)
  eval/       # metrics
  cli.py      # hfp-train-phase, hfp-eval
scripts/      # one-off analysis helpers
notebooks/    # exploration
docs/         # data access, roadmap, ethics
```

## Design principles

1. **Hormone ground truth over calendar labels** — phases and surges come from Mira LH / E3G / PdG, not period-app guesses.
2. **Leave-one-subject-out by default** — report generalization to new people, not just new cycles.
3. **No data leakage of hormone columns into features** — physiology in, hormones out.
4. **Honest scope** — commodity Fitbit + daily urine ≠ Clair’s 10-sensor / 130+ biomarker stack. This repo is the open baseline that dataset gap implies.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md). Near-term: richer Fitbit tables from mcPHASES, ovulation timing metrics, public leaderboard scripts, optional self-collect adapters (Garmin/Apple + Mira).

## Citation

If you use mcPHASES, cite the dataset authors:

> Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025). mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and Symptoms for Menstrual Health Tracking with Wearables (version 1.0.0). PhysioNet. https://doi.org/10.13026/zx6a-2c81

Please also cite/link this repo if you build on the software.

## License

MIT for code in this repository. **mcPHASES data remains under PhysioNet terms** — you must obtain it separately and comply with their DUA.

## Disclaimer

Research software only. Not a medical device. Not for contraceptive, diagnostic, or clinical decision-making.
