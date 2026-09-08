"use client";

import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
} from "recharts";
import type { UiBundle, PhaseName } from "@/lib/types";
import { pct } from "@/lib/types";
import { Methodology } from "./Methodology";

const PHASE_COLORS: Record<PhaseName, string> = {
  Menstrual: "#e89a7a",
  Follicular: "#6fd3a3",
  Fertility: "#e2b15a",
  Luteal: "#7eb6d9",
};

type Tab = "results" | "methodology";

function heat(v: number, max: number): string {
  const t = max <= 0 ? 0 : Math.min(1, v / max);
  return `rgba(111, 211, 163, ${0.08 + t * 0.75})`;
}

export function Dashboard({ data }: { data: UiBundle }) {
  const [tab, setTab] = useState<Tab>("results");
  const { phase_model: phase, dataset, lh_surge_model: lh } = data;
  const cmMax = Math.max(...phase.confusion_matrix.flat());
  const physHr = Object.entries(data.physiology_by_phase.resting_hr ?? {}).map(
    ([phaseName, stats]) => ({
      phase: phaseName,
      median: stats?.median ?? 0,
    }),
  );
  const physTemp = Object.entries(
    data.physiology_by_phase.nightly_temperature ?? {},
  ).map(([phaseName, stats]) => ({
    phase: phaseName,
    median: stats?.median ?? 0,
  }));

  const subjectSorted = [...phase.per_subject].sort((a, b) => b.accuracy - a.accuracy);

  return (
    <main>
      <header className="hero">
        <div className="eyebrow">Open research baseline · mcPHASES</div>
        <h1>Hormone from Physiology</h1>
        <p className="lede">
          Wearable resting heart rate + nightly temperature → menstrual phase and LH
          surge labels, evaluated leave-one-subject-out. Aggregates only — no raw
          PhysioNet rows ship with this UI.
        </p>
        <nav className="tabs" aria-label="Primary">
          <button
            type="button"
            className={tab === "results" ? "tab active" : "tab"}
            onClick={() => setTab("results")}
          >
            Results
          </button>
          <button
            type="button"
            className={tab === "methodology" ? "tab active" : "tab"}
            onClick={() => setTab("methodology")}
          >
            Methodology
          </button>
        </nav>
      </header>

      {tab === "methodology" ? (
        <>
          <Methodology data={data} />
          <footer className="footer">
            Generated {new Date(data.generated_at).toUTCString()}. Dataset: {dataset.name}{" "}
            v{dataset.version}. Research software only — not a medical device.
          </footer>
        </>
      ) : (
        <>
      <section className="metrics">
        <div className="metric">
          <div className="label">Phase LOSO accuracy</div>
          <div className="value">{pct(phase.accuracy)}</div>
          <div className="hint">macro-F1 {phase.macro_f1.toFixed(3)}</div>
        </div>
        <div className="metric">
          <div className="label">LH-surge LOSO accuracy</div>
          <div className="value">{pct(lh.accuracy)}</div>
          <div className="hint">macro-F1 {lh.macro_f1.toFixed(3)} (imbalanced)</div>
        </div>
        <div className="metric">
          <div className="label">Subjects / cycles</div>
          <div className="value">
            {dataset.n_subjects}
            <span style={{ fontSize: "1rem", color: "var(--muted)" }}>
              {" "}
              / {dataset.n_complete_cycles}
            </span>
          </div>
          <div className="hint">{dataset.n_days.toLocaleString()} labeled days</div>
        </div>
        <div className="metric">
          <div className="label">Wearable channels</div>
          <div className="value" style={{ fontSize: "1.25rem", marginTop: "0.55rem" }}>
            RHR · temp
          </div>
          <div className="hint">{dataset.features.length} engineered features</div>
        </div>
      </section>

      <section className="section grid-2">
        <div>
          <h2>Confusion matrix</h2>
          <p>Rows = true phase, columns = predicted (LOSO).</p>
          <div className="panel cm">
            <div className="cm-row">
              <div />
              {phase.phase_names.map((name) => (
                <div key={name} className="cm-label" style={{ textAlign: "center" }}>
                  {name.slice(0, 3)}
                </div>
              ))}
            </div>
            {phase.confusion_matrix.map((row, i) => (
              <div className="cm-row" key={phase.phase_names[i]}>
                <div className="cm-label">{phase.phase_names[i]}</div>
                {row.map((v, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="cm-cell"
                    style={{ background: heat(v, cmMax) }}
                    title={`${phase.phase_names[i]} → ${phase.phase_names[j]}: ${v}`}
                  >
                    {v}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2>Physiology by phase</h2>
          <p>Population medians from Fitbit Sense (mcPHASES).</p>
          <div className="panel" style={{ height: 220, marginBottom: "0.75rem" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={physHr}>
                <CartesianGrid stroke="rgba(232,242,238,0.08)" vertical={false} />
                <XAxis dataKey="phase" tick={{ fill: "#9bb5ab", fontSize: 11 }} />
                <YAxis tick={{ fill: "#9bb5ab", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    background: "#10241f",
                    border: "1px solid rgba(232,242,238,0.12)",
                  }}
                />
                <Bar dataKey="median" name="Resting HR" fill="#6fd3a3" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="panel" style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={physTemp}>
                <CartesianGrid stroke="rgba(232,242,238,0.08)" vertical={false} />
                <XAxis dataKey="phase" tick={{ fill: "#9bb5ab", fontSize: 11 }} />
                <YAxis tick={{ fill: "#9bb5ab", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    background: "#10241f",
                    border: "1px solid rgba(232,242,238,0.12)",
                  }}
                />
                <Bar dataKey="median" name="Nightly temp °C" fill="#e2b15a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>Hormone curves (ground truth)</h2>
        <p>
          Mira urinary LH / E3G / PdG averaged across normalized cycles — the labels the
          wearable model is trying to approach.
        </p>
        <div className="panel" style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart>
              <CartesianGrid stroke="rgba(232,242,238,0.08)" />
              <XAxis
                dataKey="cycle_pct"
                type="number"
                domain={[0, 100]}
                tick={{ fill: "#9bb5ab", fontSize: 11 }}
                label={{ value: "Cycle %", position: "insideBottom", offset: -2, fill: "#9bb5ab" }}
              />
              <YAxis tick={{ fill: "#9bb5ab", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "#10241f",
                  border: "1px solid rgba(232,242,238,0.12)",
                }}
              />
              <Line
                data={data.hormone_curves.lh}
                dataKey="mean"
                name="LH"
                stroke={PHASE_COLORS.Fertility}
                dot={false}
                strokeWidth={2}
              />
              <Line
                data={data.hormone_curves.e3g}
                dataKey="mean"
                name="E3G"
                stroke={PHASE_COLORS.Follicular}
                dot={false}
                strokeWidth={2}
              />
              <Line
                data={data.hormone_curves.pdg}
                dataKey="mean"
                name="PdG"
                stroke={PHASE_COLORS.Luteal}
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="section">
        <h2>Per-subject phase accuracy</h2>
        <p>Anonymized IDs (P01…). Shows how uneven LOSO performance is across people.</p>
        <div className="panel" style={{ overflowX: "auto", maxHeight: 360 }}>
          <table>
            <thead>
              <tr>
                <th>Subject</th>
                <th>Days</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {subjectSorted.map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.n_days}</td>
                  <td>{pct(row.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="note">
        Honest read: with only RHR + temperature (no calendar day, no hormone features),
        4-class phase LOSO accuracy is modest (~{pct(phase.accuracy)}). That gap is the
        research problem — multimodal sensing and denser pairing are what commercial
        systems claim to close. LH-surge accuracy looks high mainly from class imbalance.
      </div>

      <footer className="footer">
        Generated {new Date(data.generated_at).toUTCString()}. Dataset: {dataset.name}{" "}
        v{dataset.version} (
        <a href={`https://doi.org/${dataset.citation_doi}`} target="_blank" rel="noreferrer">
          doi:{dataset.citation_doi}
        </a>
        ). Code:{" "}
        <a
          href="https://github.com/skykauf/hormone-from-physiology"
          target="_blank"
          rel="noreferrer"
        >
          skykauf/hormone-from-physiology
        </a>
        . Research software only — not a medical device.
      </footer>
        </>
      )}
    </main>
  );
}
