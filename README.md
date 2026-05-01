# spine-usf-data

ML pipeline and FastAPI backend for predicting Adjacent Segment Disease (ASD) after lateral lumbar interbody fusion. Single-institution USF cohort (n = 546, 122 ASD events, 22.3%).

The deployed app is a Random Survival Forest (RSF) over a clinically-curated 48-source-variable feature set (419 columns after one-hot encoding of free-text surgery descriptions, then 13 continuous numerics PCA-reduced to 3 components → 61 final inputs). Honest 10-fold CV: **C = 0.614 ± 0.076**, time-dependent AUC peak at 24 months = 0.620, ECE @ 12 months = 0.057.

## Deployed app

| Layer | Stack | URL |
|---|---|---|
| Backend (this repo) | FastAPI + sksurv + DeepSurv on Render | `<https://spine-usf-data.onrender.com/docs>` |
| Frontend (separate repo) | Next.js on Vercel | `<babels.ai/spine-asd>` |


## Quick start

```bash
# Python 3.11.9 (pinned in .python-version)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the API locally on :10000
bash start.sh
# or:
uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

Health check: `GET http://localhost:10000/healthz` → `{"status":"ok", ...}`

## Repo layout

```
.
├── app.py                          # FastAPI backend (deployed to Render)
├── start.sh                        # One-line uvicorn launcher
├── requirements.txt                # Python deps (numpy<2, sksurv, pycox, shap 0.45.1, …)
├── .python-version                 # 3.11.9 (Render reads this)
│
├── Lateral Data 2014-2022.xlsx     # Original source spreadsheet (USF)
├── uncleaned_data.csv              # Excel-extracted raw CSV
├── data cleaning.ipynb             # Raw → cleaned_data.csv (numeric coercion,
│                                   #   dx flag extraction, level encoding, etc.)
├── cleaned_data.csv                # Modeling-ready dataset (546 × 116)
├── train_deepsurv.py               # Standalone DeepSurv trainer
│
├── models/
│   ├── rsf_bundle.pkl              # Deployed RSF + scaler + PCA + feature_columns
│   └── deepsurv_bundle.pkl         # Deployed DeepSurv (Cox-PH NN via pycox)
│
├── experiments/                    # Training, CV, calibration, explainability
│   ├── fix_levels_and_retrain.py   # Re-derive level columns from uncleaned_data
│   │                               #   and retrain the deployed bundles
│   ├── honest_cv.py                # Honest 10-fold stratified CV across 5 base models
│   ├── extended_models.py          # RSF/GBSA/Coxnet/CWGB/DeepSurv comparison + ensembles
│   ├── calibration_ensemble.py     # IBS, ECE @ 12/24/36/48/60 mo, isotonic calibration
│   ├── generate_paper_figures.py   # Builds fig01–fig30 manuscript figures
│   ├── shap_dependence_plots.py    # PCA × clinical interaction dependence plots
│   ├── shap_summary_publication.py # Beeswarm + global bar SHAP figures
│   ├── survshap_analysis.py        # Time-dependent SHAP (survSHAP(t)) — fig32–fig33
│   ├── survshap_rank_shift.py      # Rank-shift / early-vs-late driver — fig35–fig37
│   ├── survshap_checkpoint.npz     # Resumable per-patient checkpoint for survSHAP
│   ├── survshap_global_importance.csv          # Time-averaged |SHAP| per feature
│   ├── survshap_early_vs_late_summary.csv      # Rank shift 12mo→60mo summary
│   ├── shap_values_cohort.npz      # Cached static SHAP values
│   ├── fold_metrics.csv            # Per-fold C-index / AUC / IBS
│   ├── results.json                # honest_cv summary
│   ├── extended_results.json       # extended_models summary
│   ├── calibration_results.json    # calibration_ensemble summary
│   ├── calibration_bins_*.csv      # Calibration curve bins per model
│   ├── preset_preds.csv            # Predictions for low/med/high preset patients
│   ├── test_all_endpoints.py       # End-to-end FastAPI smoke test
│   └── test_deepsurv_endpoint.py   # DeepSurv endpoint smoke test
│
├── figures/                        # 36 manuscript-ready figures (fig01–fig37)
│
└── archive/                        # gitignored — superseded scripts/notebooks/models
                                    #   kept locally for reference; not pushed
```

## Reproducing the model from scratch

```bash
# 1. Excel → uncleaned_data.csv  (one-off; already in repo)
# 2. Raw → modeling-ready dataset
jupyter nbconvert --to notebook --execute "data cleaning.ipynb"

# 3. Patch level columns (recovers T12-L1 … L5-S1 from uncleaned_data.csv,
#    which the cleaning step had collapsed to zero) and retrain the deployed bundles
python experiments/fix_levels_and_retrain.py

# 4. Honest cross-validation across all base models
python experiments/honest_cv.py
python experiments/extended_models.py
python experiments/calibration_ensemble.py

# 5. Regenerate manuscript figures
python experiments/generate_paper_figures.py
python experiments/shap_summary_publication.py
python experiments/shap_dependence_plots.py

# 6. (optional, ~85 min) Time-dependent SHAP analysis
python -u experiments/survshap_analysis.py        # resumable via checkpoint
python experiments/survshap_rank_shift.py
```

## API reference (for external validation)

All endpoints accept JSON. Example payload schema is in `app.py` → `class PatientInput`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness check |
| `/model-info` | GET | Feature names, model version, training cohort size |
| `/predict/asd` | POST | RSF risk score + survival curve |
| `/predict/asd/deepsurv` | POST | DeepSurv risk score + survival curve |
| `/predict/asd/ensemble` | POST | Convex ensemble of RSF + DeepSurv |
| `/predict/asd/explain` | POST (SSE) | Streaming SHAP feature attributions |

### Minimal prediction example

```python
import requests
URL = "<RENDER_URL>/predict/asd"   # or http://localhost:10000

payload = {
    "Sex": 0, "Age": 65, "BMI": 28.5,
    "prior_back_surgeries": 1,
    "dx_spondylolisthesis": 1, "dx_stenosis": 1,
    "T12_L1": 0, "L1_L2": 0, "L2_L3": 0, "L3_L4": 1, "L4_L5": 1, "L5_S1": 0,
    "Perc_screws": 1, "Open": 0,
    "Average_PI": 55, "PI_LL_angle_mismatch": 12,
    "post_PI": 53, "post_PT": 22, "post_LL": 41, "post_SVA": 35,
    # … see app.py:PatientInput for the complete field list
}

r = requests.post(URL, json=payload).json()
print(r["risk_score"], r["survival_curve"][:5])
```

## External validation guidance

If validating this model:

1. Match the input schema. All field names, units, and binary encodings are documented in `app.py:PatientInput`. Spinopelvic measurements (PI, PT, SS, LL, SVA, PI-LL mismatch) must be in degrees / millimeters as defined by the SRS-Schwab convention.
2. Run predictions one of two ways:
   - Hosted API (no setup): POST patient records to the deployed Render endpoint above.
   - Local re-fit: clone the repo, run `pip install -r requirements.txt`, load `models/rsf_bundle.pkl` directly, and call `bundle["model"].predict(X)` after applying `bundle["scaler"]` and `bundle["pca"]` to the 13 continuous numerics.
3. Score w concordance-index, time-dependent AUC, and integrated Brier score @ 12/24/36/48/60 months. Reference numbers from the USF cohort (honest CV) are in `experiments/results.json` and `experiments/calibration_results.json`.
4. Known limitations are.. single-center cohort, 22% event rate, median follow-up 9 months among censored patients. Calibration is reliable at 12–24 months and degrades at 60 months. C-index drops toward chance at 48–60 months.

## Citing this work
to be added....

## License

See `LICENSE`.
