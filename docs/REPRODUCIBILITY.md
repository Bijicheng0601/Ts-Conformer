# Reproducibility and outstanding items

1. Primary predictions come from the original TS-Conformer run identified by the figure-generation script: tf_26_k3_p1_e64_a16_tt16_v16_tv16_n0p2. Original model code is included. The learnable-projector variant is separate and must not be substituted silently.
2. A raw EEG preprocessing script was located and included in `preprocessing`. Its exact correspondence to the final manuscript pipeline is unverified. A historical summary notes one recording was rerun with a different artifact-rejection threshold; confirm the actual settings before claiming exact reproduction.
3. Behavioral comparison/plotting code is included with portable paths to the shared CSVs. Raw behavioral measurement/response-label generation code and complete figure-to-file mapping still need verification; no replacement algorithm has been invented.
4. Confirm every table/figure against the final manuscript, group counts and label definitions. Additional ERP source tables are included as candidates and may exceed final manuscript scope.
5. Confirm original environment versions and full training configuration. Full retraining has not been run.
6. The author confirmed permission to publicly share de-identified behavioral and prediction data. Random codes replace source identifiers; review residual re-identification risks and choose appropriate data/code licenses with the rights holder.
7. Publish only after reviewing these items. A GitHub URL is not a DOI. Archive the approved release with Zenodo to obtain a version-specific DOI; no DOI exists yet.
