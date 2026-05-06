"""
Sensitivity analysis: exclude patients with pre-existing ASD at index
surgery (ASD B4 Surgery == 1), re-run honest 10-fold RSF CV, and compare
to the primary results.

Justification: The dictionary flags 98/546 patients (18%) as having
pre-existing adjacent-segment degeneration before the index procedure.
If the primary model's signal is contaminated by this baseline burden,
performance on the cleaner subset will diverge. If the metrics are
stable, the primary analysis is robust to the inclusion of pre-existing
cases and the manuscript can argue robustness in a single paragraph.

Outputs:
  experiments/sensitivity_no_preasd.json  — RSF metrics on the n=448 subset
  Console: side-by-side comparison vs the primary-cohort numbers in
           experiments/results.json (honest RSF) and
           experiments/calibration_results.json (RSF time-dep AUC).
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

# -----------------------------------------------------------------------------
# Load cohort, apply sensitivity filter
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

n_before = len(df)
n_pre = int((df["ASD B4 Surgery"] == 1).sum())
df = df[df["ASD B4 Surgery"] != 1].reset_index(drop=True)
n_after = len(df)
n_events = int(df[event_col].sum())
print(f"Sensitivity cohort: {n_before} -> {n_after}  (excluded {n_pre} with pre-existing ASD)")
print(f"Events in sensitivity cohort: {n_events} ({100*n_events/n_after:.1f}%)\n")

# -----------------------------------------------------------------------------
# Same feature list and pipeline as honest_cv.py
# -----------------------------------------------------------------------------
clean_feature_cols = [
    "Sex", "Age", "BMI", "prior back surgeries? (y=1)",
    "dx_adjacent_segment", "dx_spondylolisthesis", "dx_spondylosis",
    "dx_stenosis", "dx_scoliosis", "dx_flat_back", "dx_sagittal_imbalance",
    "dx_post_laminectomy", "dx_deformity",
    "Case/Type of Surgery", "Additional Procedures w/in surgery",
    "levels_fused_count",
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

numeric_like_cols = [
    "BMI", "ALIF Count", "Lateral Count",
    "Average PI", "PI-LL angle mismatch", "ABS PI-LL angle mismatch",
    "Post-op SS", "post PI", "post PT", "post LL", "post SVA",
    "length of hospital stay (d)", "levels_fused_count",
]

X_raw = df[present].copy()
for c in numeric_like_cols:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
str_cats = X_raw.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X_raw, columns=str_cats, drop_first=True).fillna(0.0)

times = df["time_surv"].astype(float).values
events = df[event_col].astype(bool).values
print(f"Encoded matrix: {X_enc.shape}\n")


def fit_pipeline(X_tr: pd.DataFrame, numeric_present: list[str]):
    binary_cols = [c for c in X_tr.columns if set(np.unique(X_tr[c])) <= {0, 1}]
    rare = [c for c in binary_cols if X_tr[c].sum() < 5]
    keep = [c for c in X_tr.columns if c not in rare]
    num_in_keep = [c for c in numeric_present if c in keep]
    scaler = StandardScaler().fit(X_tr[num_in_keep].values)
    n_pca = min(3, len(num_in_keep))
    pca = PCA(n_components=n_pca, random_state=42).fit(scaler.transform(X_tr[num_in_keep].values))

    def transform(X):
        Xk = X[keep].copy()
        pcs = pca.transform(scaler.transform(Xk[num_in_keep].values))
        rest = Xk.drop(columns=num_in_keep).astype(float).values
        return np.concatenate([rest, pcs], axis=1)
    return transform


# -----------------------------------------------------------------------------
# Honest 10-fold RSF CV with time-dep AUC at 12/24/36/48/60mo
# -----------------------------------------------------------------------------
HORIZONS = np.array([12.0, 24.0, 36.0, 48.0, 60.0])

def rsf_factory():
    return RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cis, ibss = [], []
auc_per_horizon = {h: [] for h in HORIZONS}
ibs_grid = np.percentile(times, np.linspace(10, 90, 9))

print("=== Honest RSF CV on sensitivity cohort ===")
for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
    if events[te].sum() == 0:
        cis.append(np.nan); ibss.append(np.nan)
        for h in HORIZONS: auc_per_horizon[h].append(np.nan)
        continue
    transform = fit_pipeline(X_enc.iloc[tr], numeric_like_cols)
    X_tr = transform(X_enc.iloc[tr])
    X_te = transform(X_enc.iloc[te])
    y_tr = Surv.from_arrays(event=events[tr], time=times[tr])
    y_te = Surv.from_arrays(event=events[te], time=times[te])

    m = rsf_factory(); m.fit(X_tr, y_tr)
    risk = m.predict(X_te)
    ci = concordance_index_censored(events[te], times[te], risk)[0]
    cis.append(ci)

    surv = m.predict_survival_function(X_te, return_array=False)
    S = np.zeros((len(surv), len(ibs_grid)))
    for i, fn in enumerate(surv):
        S[i] = np.interp(ibs_grid, fn.x, fn.y, left=1.0, right=fn.y[-1])
    try:
        ibs = integrated_brier_score(y_tr, y_te, S, times=ibs_grid)
    except Exception:
        ibs = np.nan
    ibss.append(ibs)

    valid_h = HORIZONS[(HORIZONS > times[te][events[te]].min()) & (HORIZONS < times[te][events[te]].max())]
    if len(valid_h):
        try:
            aucs, _ = cumulative_dynamic_auc(y_tr, y_te, risk, valid_h)
            for h, a in zip(valid_h, aucs):
                auc_per_horizon[h].append(float(a))
        except Exception:
            pass
    for h in HORIZONS:
        if h not in valid_h:
            auc_per_horizon[h].append(np.nan)
    print(f"  fold {fold}: C={ci:.3f}  IBS={ibs:.3f}")

cis = np.array(cis, dtype=float); ibss = np.array(ibss, dtype=float)
print(f"\nSensitivity RSF C-index = {np.nanmean(cis):.3f} ± {np.nanstd(cis):.3f}")
print(f"Sensitivity RSF IBS     = {np.nanmean(ibss):.3f} ± {np.nanstd(ibss):.3f}")

# -----------------------------------------------------------------------------
# Compare to primary cohort
# -----------------------------------------------------------------------------
with open(EXP / "results.json") as f: primary = json.load(f)
honest_rsf = next(r for r in primary if r["name"] == "RSF" and r["honest"])

with open(EXP / "extended_results.json") as f: ext = json.load(f)
primary_auc = {a["horizon_months"]: a["auc_mean"] for a in ext["rsf_time_auc"]}

print("\n" + "=" * 72)
print(f"{'Metric':<28} {'Primary (n=546)':>20} {'Sensitivity (n='+str(n_after)+')':>22}")
print("=" * 72)
print(f"{'C-index':<28} {honest_rsf['cindex_mean']:>15.3f} ± {honest_rsf['cindex_std']:.3f}    {np.nanmean(cis):>15.3f} ± {np.nanstd(cis):.3f}")
print(f"{'IBS':<28} {honest_rsf['ibs_mean']:>15.3f} ± {honest_rsf['ibs_std']:.3f}    {np.nanmean(ibss):>15.3f} ± {np.nanstd(ibss):.3f}")
for h in HORIZONS:
    pr = primary_auc.get(int(h), np.nan)
    se = np.nanmean(auc_per_horizon[h])
    print(f"{'AUC @ '+str(int(h))+' mo':<28} {pr:>20.3f}    {se:>20.3f}")
print("=" * 72)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out = {
    "cohort": {"n_before": n_before, "n_after": n_after, "n_excluded_pre_asd": n_pre, "n_events": n_events},
    "rsf_cindex_mean": float(np.nanmean(cis)),
    "rsf_cindex_std": float(np.nanstd(cis)),
    "rsf_cindex_per_fold": [None if np.isnan(x) else float(x) for x in cis],
    "rsf_ibs_mean": float(np.nanmean(ibss)),
    "rsf_ibs_std": float(np.nanstd(ibss)),
    "rsf_auc_per_horizon": {
        int(h): {
            "mean": float(np.nanmean(auc_per_horizon[h])),
            "std": float(np.nanstd(auc_per_horizon[h])),
        } for h in HORIZONS
    },
}
with open(EXP / "sensitivity_no_preasd.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {EXP / 'sensitivity_no_preasd.json'}")
