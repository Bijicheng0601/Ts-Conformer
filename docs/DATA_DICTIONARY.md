# Data dictionary

The manifest records every CSV column and row count.
- `subject`: random participant code. Codes match across files within a cohort; control and probiotic identifiers are separate namespaces.
- T0/T1/T2: baseline / one month / two months.
- `mean_rt_ms`: mean reaction time, milliseconds.
- `accuracy_pct`, `impulsive_rate_pct`, `omission_rate_pct`: percentages.
- `rt_variability_cv_pct`: reaction-time coefficient of variation, percent.
- `final_response_label_core2`: effective/ineffective composite behavioral response label, not a diagnostic classification.
- `true_label`: original binary response encoding (verify effective=1 against source script).
- `predicted_probability_effective`: model probability for effective response.
- `fold`: original leave-one-subject-out fold index; retained for verification.
- ERP column units and window definitions must be checked against the source analysis before release; do not infer units from amplitude alone.

Missing cells remain missing; no values were imputed during packaging. File names identify runs, not proof of correspondence to the final submitted manuscript.
