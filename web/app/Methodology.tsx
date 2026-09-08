"use client";

import type { UiBundle } from "@/lib/types";
import { pct } from "@/lib/types";

export function Methodology({ data }: { data: UiBundle }) {
  const { dataset, phase_model: phase } = data;

  return (
    <div className="methodology">
      <section className="section">
        <h2>Question</h2>
        <p>
          Can day-level wearable physiology alone recover hormone-verified menstrual
          state? This baseline answers that with public{" "}
          <a href="https://physionet.org/content/mcphases/1.0.0/" target="_blank" rel="noreferrer">
            mcPHASES
          </a>{" "}
          data — not a commercial Clair-style 130+ biomarker stack.
        </p>
      </section>

      <section className="section">
        <h2>Data</h2>
        <div className="panel method-block">
          <ul>
            <li>
              <strong>Source:</strong> mcPHASES v{dataset.version} (PhysioNet restricted
              access; DOI{" "}
              <a href={`https://doi.org/${dataset.citation_doi}`} target="_blank" rel="noreferrer">
                {dataset.citation_doi}
              </a>
              ).
            </li>
            <li>
              <strong>Cohort used here:</strong> {dataset.n_subjects} subjects,{" "}
              {dataset.n_complete_cycles} complete cycles, {dataset.n_days.toLocaleString()}{" "}
              day-level rows after filtering.
            </li>
            <li>
              <strong>Wearables:</strong> Fitbit Sense resting heart rate and nightly skin
              temperature (always). Optional mid-size tables when present in{" "}
              <code>data/raw/</code>: sleep, sleep score, HRV, respiratory rate, stress
              score, wrist temperature. Currently loaded:{" "}
              {(dataset.optional_tables_present?.length ?? 0) > 0
                ? dataset.optional_tables_present!.join(", ")
                : "none (RHR+temp only)"}.
            </li>
            <li>
              <strong>Hormone ground truth:</strong> Mira daily urine LH, E3G
              (&quot;estrogen&quot;), and PdG, plus Mira-derived phase labels (Menstrual,
              Follicular, Fertility, Luteal).
            </li>
            <li>
              <strong>What this UI shows:</strong> population aggregates and model metrics
              only. Raw participant CSVs are never shipped to Vercel (PhysioNet DUA).
            </li>
          </ul>
        </div>
      </section>

      <section className="section">
        <h2>Pipeline</h2>
        <ol className="method-steps">
          <li>
            <strong>Join</strong> hormones + RHR + nightly temperature on{" "}
            <code>id</code>, <code>day_in_study</code>.
          </li>
          <li>
            <strong>Segment cycles</strong> from menstrual-phase onset; keep complete
            cycles (all four phases, length 21–45 days).
          </li>
          <li>
            <strong>Label</strong> LH-surge days (per-person high LH quantile, floored) and
            PdG-rise days (vs early-cycle baseline) for secondary tasks.
          </li>
          <li>
            <strong>Engineer wearable features</strong> only: raw channels, 3/7-day
            rolling means, deltas, personal expanding median deviation, expanding
            z-scores, BBT-style temperature-vs-nadir, weekend flag. Optional sleep/HRV/RR
            /stress when tables are present.
          </li>
          <li>
            <strong>Train</strong> HistGradientBoosting with median imputation and{" "}
            <em>balanced</em> sample weights; majority-smooth phase predictions within
            subject (window=3) at inference only.
          </li>
          <li>
            <strong>Evaluate</strong> leave-one-subject-out (LOSO). Report balanced
            accuracy + macro-F1 for phase; biphasic luteal task; PR-AUC for LH surge.
            Channel ablations: temp_only / rhr_only / rhr_temp / multimodal.
          </li>
        </ol>
      </section>

      <section className="section">
        <h2>Leakage controls</h2>
        <div className="panel method-block">
          <p style={{ marginBottom: "0.75rem" }}>
            Features deliberately <em>exclude</em> anything derived from hormone assays or
            phase-labeled cycle position:
          </p>
          <ul>
            <li>LH, E3G, PdG (and surge/rise labels built from them)</li>
            <li>
              <code>day_in_cycle</code> / sin-cos encodings (those start from menstrual
              phase labels, so they would trivialize phase prediction)
            </li>
          </ul>
          <p style={{ marginTop: "0.75rem", marginBottom: 0 }}>
            Current feature set ({dataset.features.length}):{" "}
            {dataset.features.map((f) => (
              <code key={f} className="feat">
                {f}
              </code>
            ))}
          </p>
        </div>
      </section>

      <section className="section">
        <h2>Tasks & metrics</h2>
        <div className="grid-2">
          <div className="panel method-block">
            <h3>Phase classification</h3>
            <p>
              4-class day-level Mira phase. Primary metrics: LOSO balanced accuracy and
              macro-F1 (currently{" "}
              {pct(phase.balanced_accuracy ?? phase.accuracy)} / {phase.macro_f1.toFixed(3)}
              ). Confusion matrix and per-subject scores are on Results.
            </p>
          </div>
          <div className="panel method-block">
            <h3>Biphasic + LH surge</h3>
            <p>
              Biphasic: pre-luteal (Menstrual+Follicular) vs Luteal; Fertility excluded.
              LH surge: prefer PR-AUC / macro-F1 — raw accuracy is misleading with ~5%
              positives.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>What this is not</h2>
        <div className="panel method-block">
          <ul>
            <li>Not FDA-cleared; not for contraception, diagnosis, or treatment decisions.</li>
            <li>
              Not a recreation of proprietary multi-sensor / biomagnetic stacks — only RHR
              + temperature from mcPHASES.
            </li>
            <li>
              Not quantitative serum estradiol/progesterone in pg/mL; urine metabolites are
              the labels.
            </li>
          </ul>
        </div>
      </section>

      <section className="section">
        <h2>Reproduce</h2>
        <div className="panel method-block">
          <pre className="code-block">{`# core tables in data/raw/, then optional Fitbit extras:
export PHYSIONET_USER=... PHYSIONET_PASSWORD=...
python scripts/download_mcphases_extras.py
pip install -e ".[dev]"
python scripts/build_ui_bundle.py --data-dir data/raw --out artifacts/ui_bundle.json`}</pre>
          <p style={{ marginBottom: 0 }}>
            Code:{" "}
            <a
              href="https://github.com/skykauf/hormone-from-physiology"
              target="_blank"
              rel="noreferrer"
            >
              github.com/skykauf/hormone-from-physiology
            </a>
            . Training for this page ran on a private Tailscale host; only the aggregate
            JSON is public.
          </p>
        </div>
      </section>
    </div>
  );
}
