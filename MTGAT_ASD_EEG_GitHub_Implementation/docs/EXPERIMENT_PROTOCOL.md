# Experiment protocol

## Dataset and preprocessing
- Record dataset release, DOI/URL, access date, and usage terms.
- Verify diagnosis labels, age range, participants, channels, montage, reference, sampling rate, and event annotations.
- Document filtering, resampling, epoch length/overlap, bad-channel handling, and artifact rejection.
- Choose artifact thresholds from documented dataset-specific procedures; do not use arbitrary thresholds without justification.

## Leakage prevention
- Split participants before training/model selection. All epochs from one participant stay in one split.
- Fit normalization and feature selection using training data only.
- Use validation data for model selection; evaluate the test set once.
- For cross-cohort validation, report domain shifts and label harmonization.

## Outcomes
Report accuracy, balanced accuracy, macro-F1, sensitivity, specificity, AUROC, AUPRC, confusion matrix, and confidence intervals. AUPRC depends on class prevalence. Aggregate epoch predictions to participant-level outcomes using a rule chosen before test evaluation.

## Statistics and interpretability
Use paired predictions on the same independent test participants for model comparisons. Use a validated DeLong method for AUROC and adjust multiple comparisons (e.g., Holm). Report p-values only after actual computation. Attention is not causal evidence. Test attribution robustness and use ablations/permutation controls.

## Functional subtyping
Use participant-level embeddings. Pre-specify cluster selection; report cluster sizes, silhouette, Davies-Bouldin, repeated-run stability ARI, and independent validation. Treat subtypes as exploratory until replicated.

## Reporting
Regenerate every table and figure from actual outputs. Never represent example/synthetic values as experimental findings.
