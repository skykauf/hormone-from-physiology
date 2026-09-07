# Roadmap

## v0.1 (this release)

- [x] mcPHASES daily panel loader (hormones + RHR + nightly temperature)
- [x] Cycle segmentation + complete-cycle filter
- [x] Rolling / personal-baseline wearable features
- [x] Phase classifier + LH-surge detector with LOSO evaluation
- [x] CLI entry points + synthetic unit tests

## v0.2

- [ ] Ingest optional mcPHASES Fitbit tables (sleep, HRV, RR, SpO₂, stress)
- [ ] Ovulation / fertile-window timing metrics (day error, sensitivity/specificity)
- [ ] Ablation scripts (temp-only vs RHR-only vs multimodal)
- [ ] Reproduce figures similar to the official mcPHASES sample analysis

## v0.3

- [ ] Self-collect schema: Garmin/Apple Health export + Mira/Inito CSV adapters
- [ ] Per-person calibration / fine-tuning protocol
- [ ] Public benchmark config (fixed splits, seed, metrics JSON schema)

## Explicit non-goals (for now)

- Replicating proprietary Clair biomarkers or biomagnetic sensing
- Claiming FDA-grade quantitative hormone concentrations
- Shipping contraceptive advice or medical recommendations
