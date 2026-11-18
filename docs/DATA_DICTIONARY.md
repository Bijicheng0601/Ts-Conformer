# Data dictionary

The manifest records every CSV column and row count.
- `subject`: random participant code. Codes match across files within a cohort; control and probiotic identifiers are separate namespaces.
- T0/T1/T2: baseline / one month / two months.
- `mean_rt_ms`: mean reaction time, milliseconds.
- `accuracy_pct`, `impulsive_rate_pct`, `omission_rate_pct`: percentages.
- `rt_variability_cv_pct`: reaction-time coefficient of variation, percent.
- `final_response_label_core2`: effective/ineffective composite behavioral response label, not a diagnostic classification.
- `true_label`: binary response encoding: 0=ineffective, 1=effective.
- ERP column units and window definitions must be checked against the source analysis before release; do not infer units from amplitude alone.

Missing cells remain missing; no values were imputed during packaging. File names identify runs, not proof of correspondence to the final submitted manuscript.

Prediction CSVs contain only `true_label` and `predicted_label` (effective/ineffective). Row order and labels are preserved. Probabilities, subject codes and fold identifiers are omitted. ROC-AUC and PR-AUC cannot be recomputed from these files.
