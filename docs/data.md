# Getting mcPHASES data

## Access

1. Open https://physionet.org/content/mcphases/1.0.0/
2. Sign in / register on PhysioNet.
3. Complete the **Restricted Health Data Use Agreement**.
4. Download the release archive.

Place CSVs in `data/raw/` (gitignored) or point `MCPHASES_DATA_DIR` at the extract directory.

## Minimum tables for baselines

| File | Role |
|---|---|
| `hormones_and_selfreport.csv` | Daily LH, E3G (`estrogen`), PdG, phase, symptoms |
| `resting_heart_rate.csv` | Fitbit resting HR |
| `computed_temperature.csv` | Nightly skin temperature |

## Useful additional tables

Heart rate, HRV, sleep stages, respiratory rate, SpO₂, stress score, and Dexcom glucose — see the PhysioNet file list and the official example repo: https://github.com/chai-toronto/mcphases

## Join keys

All tables link on `id` + `day_in_study` (sleep tables may use `sleep_end_day_in_study`).

## Citation

Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025). mcPHASES (v1.0.0). PhysioNet. https://doi.org/10.13026/zx6a-2c81

Paper: https://www.nature.com/articles/s41597-026-06805-3
