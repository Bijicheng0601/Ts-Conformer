# TS-Conformer research materials — data and code

This package contains behavioral analysis tables, figure source tables, subject-level model predictions and model source code for the TS-Conformer manuscript.

## Contents
- `data/behavior`: probiotic/control ANT measures and response labels.
- `data/figure_sources`: existing behavioral, ERP and model-comparison tables.
- `data/predictions`: held-out subject predictions from identified experiment runs.
- `models`: original primary model and comparison backbones; the learnable-projector experiment is explicitly a separate variant.
- `docs`: data dictionary, limitations and release instructions.

## Access and reproducibility limits
Raw EEG, clinical records, original subject identifiers, identity mappings and trained checkpoints are excluded. Identifiers have been replaced consistently with random study codes; this is pseudonymization, not proof of anonymization. The author confirmed permission to share de-identified behavioral and prediction data. No reuse license is granted by this statement.

Training requires authorized preprocessed EEG epochs, not included here. Set `TS_DATA_ROOT` to a directory containing `ADHDtime_preprocessed` and `ADHDtime/ant_treatment_response_by_subject_core2.csv`. Model paths have been made portable; numerical training logic was preserved. Dependency versions have not been verified against the original training environment. Do not use defaults as the published configuration: consult `docs/REPRODUCIBILITY.md`.

Repository: https://github.com/Bijicheng0601/Ts-Conformer

No DOI has been assigned yet. It is not a complete raw-to-result reproduction package.

Prediction CSVs retain the original fold, pseudonymized subject, response-label and true/predicted-label fields. Only predicted_probability_effective and true_label_probability are removed where present. Labels, row order and participant codes are unchanged. ROC-AUC and PR-AUC cannot be recomputed without the probabilities.
