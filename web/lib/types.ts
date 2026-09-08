export type PhaseName = "Menstrual" | "Follicular" | "Fertility" | "Luteal";

export type UiBundle = {
  generated_at: string;
  dataset: {
    name: string;
    version: string;
    citation_doi: string;
    n_subjects: number;
    n_complete_cycles: number;
    n_days: number;
    features: string[];
    note: string;
  };
  phase_model: {
    evaluation: string;
    accuracy: number;
    macro_f1: number;
    phase_names: PhaseName[];
    confusion_matrix: number[][];
    per_subject: { id: string; n_days: number; accuracy: number }[];
  };
  lh_surge_model: {
    evaluation: string;
    accuracy: number;
    macro_f1: number;
    train_auroc_final_model?: number | null;
  };
  physiology_by_phase: Record<
    string,
    Partial<Record<PhaseName, { n: number; median: number; q25: number; q75: number; mean: number }>>
  >;
  hormone_curves: Record<
    string,
    { cycle_pct: number; mean: number; sem: number; n: number }[]
  >;
  phase_counts: Record<string, number>;
};

export function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}
