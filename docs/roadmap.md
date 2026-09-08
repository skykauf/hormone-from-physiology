# Roadmap

## v0.1 (this release)

- [x] mcPHASES daily panel loader (hormones + RHR + nightly temperature)
- [x] Cycle segmentation + complete-cycle filter
- [x] Rolling / personal-baseline wearable features
- [x] Phase classifier + LH-surge detector with LOSO evaluation
- [x] CLI entry points + synthetic unit tests

## v0.2

- [x] Ingest optional mcPHASES Fitbit tables (sleep, HRV, RR, stress, wrist temp) when present
- [x] Stronger features (z-scores, temp-vs-nadir) + class-balanced LOSO + phase smoothing
- [x] Biphasic luteal task + PR-AUC for LH surge
- [x] Ablation harness (temp / RHR / RHR+temp / multimodal)
- [ ] Ovulation / fertile-window timing metrics (day error, sensitivity/specificity)
- [ ] Download optional Fitbit tables into `data/raw/` for true multimodal lift

## v0.3

- [ ] Self-collect schema: Garmin/Apple Health export + Mira/Inito CSV adapters
- [ ] Per-person calibration / fine-tuning protocol
- [ ] Public benchmark config (fixed splits, seed, metrics JSON schema)

## Explicit non-goals (for now)

- Replicating proprietary Clair biomarkers or biomagnetic sensing
- Claiming FDA-grade quantitative hormone concentrations
- Shipping contraceptive advice or medical recommendations
