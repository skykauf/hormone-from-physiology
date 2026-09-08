export type PhaseName = "Menstrual" | "Follicular" | "Fertility" | "Luteal";

export type AblationConfig = {
  n_features: number;
  features: string[];
  phase: {
    accuracy: number;
    balanced_accuracy?: number;
    macro_f1: number;
    per_class_recall?: Record<string, number>;
    confusion_matrix?: number[][];
    phase_names?: PhaseName[];
    per_subject_median_accuracy?: number;
  };
  biphasic: {
    accuracy?: number;
    balanced_accuracy?: number;
    macro_f1?: number;
    pr_auc?: number;
  };
  lh_surge: {
    accuracy?: number;
    balanced_accuracy?: number;
    macro_f1?: number;
    pr_auc?: number;
    positive_rate?: number;
    sensitivity_at_fpr?: { fpr_target: number; fpr: number; sensitivity: number };
  };
};

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
    channel_set?: string;
    optional_tables_present?: string[];
    note: string;
  };
  phase_model: {
    evaluation: string;
    accuracy: number;
    balanced_accuracy?: number;
    macro_f1: number;
    per_class_recall?: Record<string, number>;
    phase_names: PhaseName[];
    confusion_matrix: number[][];
    per_subject: { id: string; n_days: number; accuracy: number }[];
  };
  biphasic_model?: {
    evaluation: string;
    definition: string;
    accuracy?: number;
    balanced_accuracy?: number;
    macro_f1?: number;
    pr_auc?: number;
  };
  lh_surge_model: {
    evaluation: string;
    accuracy: number;
    balanced_accuracy?: number;
    macro_f1: number;
    pr_auc?: number | null;
    positive_rate?: number;
    sensitivity_at_fpr?: { fpr_target: number; fpr: number; sensitivity: number };
    train_auroc_final_model?: number | null;
  };
  ablations?: {
    optional_tables_present: string[];
    configs: Record<string, AblationConfig>;
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

export function pct(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}
