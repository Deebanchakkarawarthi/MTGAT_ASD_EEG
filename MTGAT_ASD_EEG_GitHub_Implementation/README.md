# MT-GAT for EEG-Based ASD Analysis

Research code scaffold for **Interpretable Multi-Task Graph Attention Networks for Latent Phenotype Profiling, State Tracking, and Functional Subtyping in ASD**.

## Included
- EEG loading, filtering, resampling, and fixed-length epoching
- Frequency-band PLV connectivity
- Differentiable dynamic phase-synchrony weighting module
- Graph attention, GRU temporal encoder, and FiLM conditioning
- ASD classification training utilities and evaluation metrics
- Exploratory clustering and plotting utilities
- Tests, experiment protocol, configuration, and GitHub Actions

## Scientific safeguards
- No experimental results are included or fabricated.
- Do not commit private EEG data or data downloaded under restricted terms.
- Split at participant level to prevent leakage.
- Phenotype/severity and state heads require verified labels.
- PLV reflects phase synchrony, not causal or anatomical connectivity.
- Functional subtyping is exploratory until independently validated.

## Install
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/check_environment.py
```

## Workflow
1. Configure `configs/default.yaml`.
2. Place permitted EEG files under `data/raw/`.
3. Extract epochs and PLV:
   `python scripts/extract_features.py --config configs/default.yaml`
4. Create a verified label CSV and model-ready sequence tensors. See `docs/EXPERIMENT_PROTOCOL.md` and `docs/MODEL_NOTES.md`.
5. Train using `scripts/train_example.py` only after creating tensors with the documented shapes.

The feature extractor creates one graph per epoch. The temporal model expects consecutive epochs grouped into sequences, so this grouping and label harmonization must be completed explicitly for each dataset. The example trainer optimizes the ASD head only; multi-task training requires verified labels and corresponding loss terms.

## Suggested repository structure
`configs/`, `data/`, `docs/`, `mtgat/`, `scripts/`, `tests/`, `.github/workflows/`.

Record dataset version, split IDs, random seed, package versions, exclusions, and preprocessing choices. Never report illustrative values as measured results.
