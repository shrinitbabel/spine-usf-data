"""
Honest CV experiment — answers two questions:

  1. How much of the reported C ≈ 0.60 was data leakage?
     (PCA + scaler + rare-col drop are currently fit on the FULL dataset
      before StratifiedKFold splits. We re-run those steps inside each
      fold and compare.)

  2. Holding the feature set fixed (no reduction), does ANY model in
     scikit-survival meaningfully separate the 8 clinical preset
     patients? RSF / GBSA / Coxnet, all 10-fold CV, all honest.

Outputs:
    experiments/results.json     — per-model CV metrics
    experiments/preset_preds.csv — predicted survival on the 8 presets
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments"
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load & label (matches the v1.2 notebook exactly)
# -----------------------------------------------------------------------------

df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)

event_col = "REVERIFIED ASD"
df[event_col] = df[event_col].fillna(0).astype(int)
df["time_surv"] = np.where(
    df[event_col] == 1,
    df["Time Until ASD Diagnosis (months)"],
    df["Time Without_ASD (months)"],
)
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)

print(f"Cohort: {len(df)} patients, {int(df[event_col].sum())} events")

# -----------------------------------------------------------------------------
# 2. Feature columns — the same list the v1.2 model was trained on
# -----------------------------------------------------------------------------

clean_feature_cols = [
    "Sex", "Age", "BMI", "prior back surgeries? (y=1)",
    "dx_adjacent_segment", "dx_spondylolisthesis", "dx_spondylosis",
    "dx_stenosis", "dx_scoliosis", "dx_flat_back", "dx_sagittal_imbalance",
    "dx_post_laminectomy", "dx_deformity",
    "Case/Type of Surgery", "Additional Procedures w/in surgery",
    "levels_fused_count",   # Variant C: single integer instead of 6 binaries
    "Perc screws?", "Open", "Open Check V2", "Standalone XLIF Check",
    "Retroperitoneal Approach (LLIF ± ALIF)", "Anterior + Posterior Apporoach",
    "Osteotomies (yes/no)", "osteotomy level",
    "ALIF Count", "Lateral Count", "ACR (y=1)", "ACR level",
    "Average PI", "PI-LL angle mismatch", "ABS PI-LL angle mismatch",
    "PI-LL Mismatch Category (1 = mismatch > +/- 9",
    "PI-LL Mismatch Category (1 = mismatch > +/- 10",
    "(1 = PI>50)",
    "Post-op SS", "post PI", "post PT", "post LL", "post SVA",
    "infection 1=yes", "DVT  1=yes", "PE  1=yes", "MI 1=yes",
    "femoral palsy (knee extension weakness) 1=yes",
    "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute thigh paresthesia (immediate post op)", "psoas hematoma",
    "length of hospital stay (d)",
]
present = [c for c in clean_feature_cols if c in df.columns]
missing = [c for c in clean_feature_cols if c not in df.columns]
if missing:
    print(f"WARN: {len(missing)} expected columns missing from CSV: {missing[:3]}…")

numeric_like_cols = [
    "BMI", "ALIF Count", "Lateral Count",
    "Average PI", "PI-LL angle mismatch", "ABS PI-LL angle mismatch",
    "Post-op SS", "post PI", "post PT", "post LL", "post SVA",
    "length of hospital stay (d)", "levels_fused_count",
]

# -----------------------------------------------------------------------------
# 3. One-shot raw frame (numeric coercion + one-hot of string cats only)
#    NB: rare-col drop, scaling, PCA all happen per-fold below.
# -----------------------------------------------------------------------------

X_raw = df[present].copy()
for c in numeric_like_cols:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
str_cats = X_raw.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X_raw, columns=str_cats, drop_first=True).fillna(0.0)

times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
print(f"Encoded matrix: {X_enc.shape}, str cats one-hot expanded: {str_cats}")


# -----------------------------------------------------------------------------
# 4. Pipelines — leaky (current notebook) vs honest (refit per fold)
# -----------------------------------------------------------------------------

def fit_pipeline(X_tr: pd.DataFrame, numeric_present: list[str]):
    """Fit rare-col drop + scaler + PCA on training fold only. Returns transform fn."""
    binary_cols = [c for c in X_tr.columns if set(np.unique(X_tr[c])) <= {0, 1}]
    rare = [c for c in binary_cols if X_tr[c].sum() < 5]
    keep = [c for c in X_tr.columns if c not in rare]

    num_in_keep = [c for c in numeric_present if c in keep]
    scaler = StandardScaler().fit(X_tr[num_in_keep].values)
    n_pca = min(3, len(num_in_keep))
    pca = PCA(n_components=n_pca, random_state=42).fit(scaler.transform(X_tr[num_in_keep].values))

    def transform(X: pd.DataFrame) -> np.ndarray:
        Xk = X[keep].copy()
        pcs = pca.transform(scaler.transform(Xk[num_in_keep].values))
        rest = Xk.drop(columns=num_in_keep).astype(float).values
        return np.concatenate([rest, pcs], axis=1)

    return transform


def cv_eval(model_factory, *, honest: bool, name: str, n_splits: int = 10) -> dict:
    """Cross-validated C-index + IBS for one model spec."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if not honest:
        # leaky: fit preprocessing once on the full dataset
        transform = fit_pipeline(X_enc, numeric_like_cols)
        X_full = transform(X_enc)

    cis, ibss = [], []
    time_grid = np.percentile(times, np.linspace(10, 90, 9))

    for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
        if events[te].sum() == 0:
            cis.append(np.nan); ibss.append(np.nan); continue

        if honest:
            transform = fit_pipeline(X_enc.iloc[tr], numeric_like_cols)
            X_tr = transform(X_enc.iloc[tr])
            X_te = transform(X_enc.iloc[te])
        else:
            X_tr, X_te = X_full[tr], X_full[te]

        y_tr = Surv.from_arrays(event=events[tr], time=times[tr])
        y_te = Surv.from_arrays(event=events[te], time=times[te])

        try:
            model = model_factory()
            model.fit(X_tr, y_tr)
            risk = model.predict(X_te)
            ci = concordance_index_censored(events[te], times[te], risk)[0]
        except Exception as ex:
            print(f"  [{name}] fold {fold} fit error: {ex}")
            cis.append(np.nan); ibss.append(np.nan); continue

        try:
            surv = model.predict_survival_function(X_te, return_array=False)
            S = np.zeros((len(surv), len(time_grid)))
            for i, fn in enumerate(surv):
                S[i] = np.interp(time_grid, fn.x, fn.y, left=1.0, right=fn.y[-1])
            ibs = integrated_brier_score(y_tr, y_te, S, times=time_grid)
        except Exception as ex:
            ibs = np.nan

        cis.append(ci); ibss.append(ibs)

    cis = np.array(cis, dtype=float); ibss = np.array(ibss, dtype=float)
    return {
        "name": name,
        "honest": honest,
        "cindex_mean": float(np.nanmean(cis)),
        "cindex_std": float(np.nanstd(cis)),
        "cindex_per_fold": [None if np.isnan(x) else float(x) for x in cis],
        "ibs_mean": float(np.nanmean(ibss)),
        "ibs_std": float(np.nanstd(ibss)),
    }


# -----------------------------------------------------------------------------
# 5. Models to compare (all features kept, per user instruction)
# -----------------------------------------------------------------------------

def rsf_factory():
    return RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )

def gbsa_factory():
    return GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
    )

def coxnet_factory():
    # tiny l1_ratio, mild alpha — reproducible single-model fit
    return CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=20, fit_baseline_model=True)


# -----------------------------------------------------------------------------
# 6. Run
# -----------------------------------------------------------------------------

results = []

print("\n=== RSF — leaky (current notebook approach) ===")
results.append(cv_eval(rsf_factory, honest=False, name="RSF"))
print(f"  C = {results[-1]['cindex_mean']:.3f} ± {results[-1]['cindex_std']:.3f}")

print("\n=== RSF — honest (refit per fold) ===")
results.append(cv_eval(rsf_factory, honest=True, name="RSF"))
print(f"  C = {results[-1]['cindex_mean']:.3f} ± {results[-1]['cindex_std']:.3f}")

print("\n=== GBSA — honest ===")
results.append(cv_eval(gbsa_factory, honest=True, name="GBSA"))
print(f"  C = {results[-1]['cindex_mean']:.3f} ± {results[-1]['cindex_std']:.3f}")

print("\n=== Coxnet — honest ===")
results.append(cv_eval(coxnet_factory, honest=True, name="Coxnet"))
print(f"  C = {results[-1]['cindex_mean']:.3f} ± {results[-1]['cindex_std']:.3f}")

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nSaved -> {OUT_DIR / 'results.json'}")

# -----------------------------------------------------------------------------
# 7. Score the 8 clinical presets on each finalized (full-data) model
#    so we can see how much each model spreads them.
# -----------------------------------------------------------------------------

print("\n=== Scoring 8 clinical presets ===")

# Build a single-row payload generator that mirrors what the API does.
PRESETS = {
    "Low risk (short fusion)": dict(
        Sex=1, Age=50, BMI=24, prior_back_surgeries=0,
        dx_stenosis=1,
        L4_L5=1,
        Open=0, Perc_screws=1, Standalone_XLIF_Check=1, Retroperitoneal_Approach=1,
        Lateral_Count=0, ALIF_Count=0,
        Average_PI=48, PI_LL=4, ABS_PI_LL=4, PI_gt_50=0,
        post_LL=50, post_SVA=10, post_PI=48, post_PT=13, Post_op_SS=35,
        LOS=2,
    ),
    "Adjacent stress": dict(
        Sex=1, Age=63, BMI=28, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=0, Lateral_Count=1, ALIF_Count=0,
        Average_PI=58, PI_LL=15, ABS_PI_LL=15, PI_gt_50=1,
        post_LL=45, post_SVA=35, post_PI=58, post_PT=22, Post_op_SS=28,
        LOS=4,
    ),
    "Deformity / long construct": dict(
        Sex=1, Age=75, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_scoliosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1, AP_approach=1, ACR=1,
        Lateral_Count=3, ALIF_Count=0,
        Average_PI=65, PI_LL=25, ABS_PI_LL=25, PI_gt_50=1,
        post_LL=40, post_SVA=65, post_PI=65, post_PT=30, Post_op_SS=20,
        LOS=8,
    ),
    "Single-level spondy": dict(
        Sex=1, Age=60, BMI=27, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylolisthesis=1,
        L4_L5=1,
        Open=1, Lateral_Count=1, ALIF_Count=0,
        Average_PI=52, PI_LL=10, ABS_PI_LL=10, PI_gt_50=1,
        post_LL=45, post_SVA=25, post_PI=52, post_PT=16, Post_op_SS=30,
        LOS=3,
    ),
    "Revision lumbar": dict(
        Sex=1, Age=68, BMI=30, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=1, Osteotomies=1, AP_approach=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=56, PI_LL=18, ABS_PI_LL=18, PI_gt_50=1,
        post_LL=45, post_SVA=40, post_PI=56, post_PT=24, Post_op_SS=27,
        LOS=5,
    ),
    "Flat-back correction": dict(
        Sex=1, Age=71, BMI=29, prior_back_surgeries=1,
        dx_stenosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1, AP_approach=1, ACR=1,
        Lateral_Count=2, ALIF_Count=0,
        Average_PI=60, PI_LL=25, ABS_PI_LL=25, PI_gt_50=1,
        post_LL=42, post_SVA=55, post_PI=60, post_PT=28, Post_op_SS=22,
        LOS=8,
    ),
    "MIS short fusion": dict(
        Sex=1, Age=55, BMI=26, prior_back_surgeries=0,
        dx_stenosis=1,
        L4_L5=1,
        Open=0, Perc_screws=1, Standalone_XLIF_Check=1, Retroperitoneal_Approach=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=50, PI_LL=6, ABS_PI_LL=6, PI_gt_50=1,
        post_LL=48, post_SVA=18, post_PI=50, post_PT=14, Post_op_SS=33,
        LOS=2,
    ),
    "Elderly mismatch": dict(
        Sex=1, Age=78, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1, dx_sagittal_imbalance=1,
        L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=62, PI_LL=22, ABS_PI_LL=22, PI_gt_50=1,
        post_LL=45, post_SVA=50, post_PI=62, post_PT=26, Post_op_SS=25,
        LOS=6,
    ),
}

# Map preset short keys → CSV column names
KEY_TO_COL = {
    "Sex": "Sex", "Age": "Age", "BMI": "BMI",
    "prior_back_surgeries": "prior back surgeries? (y=1)",
    "dx_stenosis": "dx_stenosis", "dx_spondylolisthesis": "dx_spondylolisthesis",
    "dx_spondylosis": "dx_spondylosis", "dx_scoliosis": "dx_scoliosis",
    "dx_flat_back": "dx_flat_back", "dx_sagittal_imbalance": "dx_sagittal_imbalance",
    "T12_L1": "T12-L1", "L1_L2": "L1-L2", "L2_L3": "L2-L3",
    "L3_L4": "L3-L4", "L4_L5": "L4-L5", "L5_S1": "L5-S1",
    "Open": "Open", "Perc_screws": "Perc screws?",
    "Standalone_XLIF_Check": "Standalone XLIF Check",
    "Retroperitoneal_Approach": "Retroperitoneal Approach (LLIF ± ALIF)",
    "AP_approach": "Anterior + Posterior Apporoach",
    "Osteotomies": "Osteotomies (yes/no)",
    "Lateral_Count": "Lateral Count", "ALIF_Count": "ALIF Count",
    "ACR": "ACR (y=1)",
    "Average_PI": "Average PI",
    "PI_LL": "PI-LL angle mismatch",
    "ABS_PI_LL": "ABS PI-LL angle mismatch",
    "PI_gt_50": "(1 = PI>50)",
    "post_LL": "post LL", "post_SVA": "post SVA",
    "post_PI": "post PI", "post_PT": "post PT",
    "Post_op_SS": "Post-op SS",
    "LOS": "length of hospital stay (d)",
}

def preset_to_row(preset: dict) -> pd.DataFrame:
    row = {c: 0 for c in X_enc.columns}
    for k, v in preset.items():
        col = KEY_TO_COL.get(k)
        if col and col in row:
            row[col] = v
    return pd.DataFrame([row])[X_enc.columns]


# Fit each model on the FULL dataset using honest pipeline (refit on 100% of data).
final_transform = fit_pipeline(X_enc, numeric_like_cols)
X_full_honest = final_transform(X_enc)
y_full = Surv.from_arrays(event=events, time=times)

preset_rows = [(name, final_transform(preset_to_row(p))) for name, p in PRESETS.items()]

preset_records = []
for model_name, factory in [("RSF", rsf_factory), ("GBSA", gbsa_factory), ("Coxnet", coxnet_factory)]:
    try:
        m = factory()
        m.fit(X_full_honest, y_full)
        cohort_risk = m.predict(X_full_honest)
        for pname, x in preset_rows:
            r = float(m.predict(x)[0])
            pct = float((cohort_risk < r).mean() * 100)
            # 5y survival probability
            try:
                sf = m.predict_survival_function(x, return_array=False)[0]
                p5y = float(np.interp(60, sf.x, sf.y))
            except Exception:
                p5y = float("nan")
            preset_records.append({
                "model": model_name, "preset": pname,
                "risk_score": r, "percentile_vs_cohort": pct,
                "asd_free_5y": p5y,
            })
    except Exception as ex:
        print(f"  [{model_name}] failed on presets: {ex}")

pred_df = pd.DataFrame(preset_records)
pred_df.to_csv(OUT_DIR / "preset_preds.csv", index=False)

# Summary
if not pred_df.empty:
    print("\n=== Preset spread per model ===")
    for m in pred_df["model"].unique():
        sub = pred_df[pred_df["model"] == m]
        spread = sub["asd_free_5y"].max() - sub["asd_free_5y"].min()
        ranked = sub.sort_values("percentile_vs_cohort")["preset"].tolist()
        print(f"  {m}: 5y range = {sub['asd_free_5y'].min():.2f}–{sub['asd_free_5y'].max():.2f} "
              f"(spread {spread*100:.1f} pts)")
        print(f"      ranking low->high: {ranked}")

print(f"\nSaved -> {OUT_DIR / 'preset_preds.csv'}")
