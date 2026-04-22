"""
Calibration + ensemble vs DeepSurv-alone fidelity check.

For each fold (10-fold honest CV):
  - Fit RSF and DeepSurv on the training fold (PCA refit per fold).
  - Get test-fold risk scores AND survival probabilities at horizons
    (12, 24, 36, 48, 60 months).
  - Ensemble survival = mean of RSF and DeepSurv survival functions.

Then for each model (RSF / DeepSurv / Ensemble):
  - Per-fold C-index, IBS, time-AUC at each horizon (variance / stability).
  - Pooled ECE at each horizon (calibration), using KM observed survival
    within each prediction-bin (matches the ECE method in
    `try copy 3_model_development.ipynb`).
  - 36-month classification metrics (AUC, precision, recall, F1) at the
    cohort-relative top-tertile risk cutoff.

Outputs:
  experiments/calibration_results.json
  experiments/calibration_bins_<model>.csv  (per-bin observed vs predicted)
"""

from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from lifelines import KaplanMeierFitter

import torch
import torchtuples as tt
from pycox.models import CoxPH as CoxPH_NN

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Load (matches prior experiments)
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
df["REVERIFIED ASD"] = df["REVERIFIED ASD"].fillna(0).astype(int)
df["time_surv"] = np.where(
    df["REVERIFIED ASD"] == 1,
    df["Time Until ASD Diagnosis (months)"],
    df["Time Without_ASD (months)"],
)
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)

clean_feature_cols = [
    "Sex","Age","BMI","prior back surgeries? (y=1)",
    "dx_adjacent_segment","dx_spondylolisthesis","dx_spondylosis","dx_stenosis",
    "dx_scoliosis","dx_flat_back","dx_sagittal_imbalance","dx_post_laminectomy","dx_deformity",
    "Case/Type of Surgery","Additional Procedures w/in surgery",
    "T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1",
    "Perc screws?","Open","Open Check V2","Standalone XLIF Check",
    "Retroperitoneal Approach (LLIF ± ALIF)","Anterior + Posterior Apporoach",
    "Osteotomies (yes/no)","osteotomy level",
    "ALIF Count","Lateral Count","ACR (y=1)","ACR level",
    "Average PI","PI-LL angle mismatch","ABS PI-LL angle mismatch",
    "PI-LL Mismatch Category (1 = mismatch > +/- 9",
    "PI-LL Mismatch Category (1 = mismatch > +/- 10","(1 = PI>50)",
    "Post-op SS","post PI","post PT","post LL","post SVA",
    "infection 1=yes","DVT  1=yes","PE  1=yes","MI 1=yes",
    "femoral palsy (knee extension weakness) 1=yes",
    "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute thigh paresthesia (immediate post op)","psoas hematoma",
    "length of hospital stay (d)",
]
present = [c for c in clean_feature_cols if c in df.columns]
numeric_like = [
    "BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
    "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
    "post SVA","length of hospital stay (d)",
]

X_raw = df[present].copy()
for c in numeric_like:
    if c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
str_cats = X_raw.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X_raw, columns=str_cats, drop_first=True).fillna(0.0)
times_all = df["time_surv"].astype(float).values
events_all = df["REVERIFIED ASD"].astype(bool).values
N = len(df)

print(f"Cohort: {N} patients, {int(events_all.sum())} events, {X_enc.shape[1]} encoded cols")

# --------------------------------------------------------------------------
# Per-fold preprocessing pipeline
# --------------------------------------------------------------------------
def fit_pipeline(X_tr, numeric_present):
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


# --------------------------------------------------------------------------
# Survival evaluators
# --------------------------------------------------------------------------
HORIZONS = [12.0, 24.0, 36.0, 48.0, 60.0]

def rsf_surv_at(model, X, horizons):
    """Return (n_X, n_horizons) survival matrix from sksurv model."""
    funcs = model.predict_survival_function(X, return_array=False)
    out = np.zeros((len(funcs), len(horizons)))
    for i, fn in enumerate(funcs):
        out[i] = np.interp(horizons, fn.x, fn.y, left=1.0, right=fn.y[-1])
    return out

def deepsurv_surv_at(model, X, horizons):
    """Return (n_X, n_horizons) survival matrix from pycox CoxPH."""
    surv_df = model.predict_surv_df(X.astype("float32"))
    # surv_df: rows = times, cols = samples
    out = np.zeros((X.shape[0], len(horizons)))
    grid = surv_df.index.values.astype(float)
    for j in range(X.shape[0]):
        sj = surv_df.iloc[:, j].values
        out[j] = np.interp(horizons, grid, sj, left=1.0, right=sj[-1])
    return out

def fit_deepsurv(Xtr, t_tr, e_tr):
    in_features = Xtr.shape[1]
    net = tt.practical.MLPVanilla(
        in_features=in_features, num_nodes=[32, 16], out_features=1,
        batch_norm=True, dropout=0.5, output_bias=False,
    )
    model = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
    y_tr = (t_tr.astype("float32"), e_tr.astype("float32"))
    model.fit(Xtr.astype("float32"), y_tr, batch_size=64, epochs=80, verbose=False)
    model.compute_baseline_hazards()
    return model


# --------------------------------------------------------------------------
# CV: collect per-fold predictions
# --------------------------------------------------------------------------
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# OOF storage: risk score + survival at horizons
OOF = {
    "RSF":      {"risk": np.full(N, np.nan), "S": np.full((N, len(HORIZONS)), np.nan)},
    "DeepSurv": {"risk": np.full(N, np.nan), "S": np.full((N, len(HORIZONS)), np.nan)},
    "Ensemble": {"risk": np.full(N, np.nan), "S": np.full((N, len(HORIZONS)), np.nan)},
}
fold_metrics = []   # per-fold C, IBS, AUC@horizons for each model

for fold, (tr, te) in enumerate(skf.split(X_enc.values, events_all.astype(int)), 1):
    print(f"--- Fold {fold} ---")
    if events_all[te].sum() == 0:
        print("  (no events, skipping)"); continue
    transform = fit_pipeline(X_enc.iloc[tr], numeric_like)
    Xtr, Xte = transform(X_enc.iloc[tr]), transform(X_enc.iloc[te])
    ytr = Surv.from_arrays(event=events_all[tr], time=times_all[tr])
    yte = Surv.from_arrays(event=events_all[te], time=times_all[te])

    # ---- RSF
    rsf_m = RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    ).fit(Xtr, ytr)
    risk_rsf = rsf_m.predict(Xte)
    S_rsf = rsf_surv_at(rsf_m, Xte, HORIZONS)
    OOF["RSF"]["risk"][te] = risk_rsf
    OOF["RSF"]["S"][te] = S_rsf

    # ---- DeepSurv
    try:
        ds_m = fit_deepsurv(Xtr, times_all[tr], events_all[tr])
        risk_ds = ds_m.predict(Xte.astype("float32")).reshape(-1)
        S_ds = deepsurv_surv_at(ds_m, Xte, HORIZONS)
        OOF["DeepSurv"]["risk"][te] = risk_ds
        OOF["DeepSurv"]["S"][te] = S_ds
    except Exception as ex:
        print(f"  DeepSurv failed: {ex}")
        risk_ds = None; S_ds = None

    # ---- Ensemble: rank-avg risks, mean of survival functions
    if risk_ds is not None:
        r_rank = (pd.Series(risk_rsf).rank().values + pd.Series(risk_ds).rank().values) / 2
        S_ens = (S_rsf + S_ds) / 2
        OOF["Ensemble"]["risk"][te] = r_rank
        OOF["Ensemble"]["S"][te] = S_ens
    else:
        S_ens = None

    # ---- Per-fold metrics for each model
    fold_row = {"fold": fold}
    t_max = min(times_all[tr].max(), times_all[te].max()) - 1e-3
    t_min = max(times_all[tr].min(), times_all[te].min()) + 1e-3
    valid_h = [h for h in HORIZONS if t_min < h < t_max]

    # Build a small IBS time-grid restricted to test-fold range (sksurv requires this)
    ibs_grid = np.linspace(t_min, t_max, 10)

    def metrics_for(name, risk, S):
        if risk is None: return None
        ci = concordance_index_censored(events_all[te], times_all[te], risk)[0]
        # IBS needs survival at the IBS grid
        if name == "RSF":
            S_ibs = rsf_surv_at(rsf_m, Xte, list(ibs_grid))
        elif name == "DeepSurv":
            S_ibs = deepsurv_surv_at(ds_m, Xte, list(ibs_grid))
        else:
            S_rsf_g = rsf_surv_at(rsf_m, Xte, list(ibs_grid))
            S_ds_g  = deepsurv_surv_at(ds_m, Xte, list(ibs_grid))
            S_ibs = (S_rsf_g + S_ds_g) / 2
        try:
            ibs = integrated_brier_score(ytr, yte, S_ibs, times=ibs_grid)
        except Exception:
            ibs = float("nan")
        try:
            aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, np.asarray(valid_h, dtype=float))
            aucs = np.atleast_1d(aucs)
            auc_dict = {f"auc_{int(h)}": float(a) for h, a in zip(valid_h, aucs)}
        except Exception:
            auc_dict = {}
        out = {f"{name}_cindex": ci, f"{name}_ibs": ibs}
        out.update({f"{name}_{k}": v for k, v in auc_dict.items()})
        return out

    for name, risk, S in [("RSF", risk_rsf, S_rsf), ("DeepSurv", risk_ds, S_ds), ("Ensemble", risk_rsf if S_ens is None else None, S_ens)]:
        # for Ensemble use the rank-avg risks
        if name == "Ensemble":
            risk = OOF["Ensemble"]["risk"][te] if S_ens is not None else None
        m = metrics_for(name, risk, S)
        if m: fold_row.update(m)

    fold_metrics.append(fold_row)
    print(f"  C: RSF={fold_row.get('RSF_cindex',float('nan')):.3f}  "
          f"DeepSurv={fold_row.get('DeepSurv_cindex',float('nan')):.3f}  "
          f"Ens={fold_row.get('Ensemble_cindex',float('nan')):.3f}")

print("\nFolds done.\n")

fmdf = pd.DataFrame(fold_metrics)
fmdf.to_csv(OUT / "fold_metrics.csv", index=False)


# --------------------------------------------------------------------------
# Per-fold stability summary (mean / std / range)
# --------------------------------------------------------------------------
def stability(model_name):
    cols = [c for c in fmdf.columns if c.startswith(model_name + "_")]
    out = {}
    for c in cols:
        v = fmdf[c].dropna().values
        if len(v) == 0: continue
        out[c] = {
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "range": float(np.max(v) - np.min(v)),
        }
    return out

stability_summary = {m: stability(m) for m in ["RSF", "DeepSurv", "Ensemble"]}

print("=== Per-fold stability (mean +/- std [min, max]) ===")
for metric in ["cindex", "ibs", "auc_24", "auc_36", "auc_60"]:
    print(f"\n  {metric}:")
    for m in ["RSF", "DeepSurv", "Ensemble"]:
        key = f"{m}_{metric}"
        if key in stability_summary[m]:
            s = stability_summary[m][key]
            print(f"    {m:9s}  {s['mean']:.3f} +/- {s['std']:.3f}   [{s['min']:.3f}, {s['max']:.3f}]")


# --------------------------------------------------------------------------
# Pooled OOF C-index for each model (single concordance over all 546)
# --------------------------------------------------------------------------
print("\n=== OOF concordance (all 546 patients pooled) ===")
oof_cindex = {}
for name in ("RSF", "DeepSurv", "Ensemble"):
    risk = OOF[name]["risk"]
    valid = ~np.isnan(risk)
    ci = concordance_index_censored(events_all[valid], times_all[valid], risk[valid])[0]
    oof_cindex[name] = ci
    print(f"  {name:9s}  C = {ci:.3f}  (n={int(valid.sum())})")


# --------------------------------------------------------------------------
# Pooled ECE per horizon (Kaplan-Meier observed in each prediction bin)
# --------------------------------------------------------------------------
def pooled_ece(name, horizon_idx, n_bins=10):
    S = OOF[name]["S"][:, horizon_idx]
    valid = ~np.isnan(S)
    preds = S[valid]
    t = times_all[valid]; e = events_all[valid]
    if len(preds) < n_bins * 2:
        return None
    # Bin by predicted survival quantile
    bin_id = pd.qcut(pd.Series(preds).rank(method="first"),
                     q=n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(np.unique(bin_id)):
        mask = bin_id == b
        if mask.sum() < 3: continue
        try:
            km = KaplanMeierFitter().fit(t[mask], event_observed=e[mask])
            obs = float(km.predict(HORIZONS[horizon_idx]))
        except Exception:
            obs = float("nan")
        rows.append({
            "bin": int(b), "n": int(mask.sum()),
            "pred_mean": float(np.mean(preds[mask])), "obs": obs,
        })
    if not rows: return None
    bdf = pd.DataFrame(rows).dropna(subset=["obs"])
    counts = bdf["n"].values.astype(float)
    ece = float(np.sum((counts / counts.sum()) * np.abs(bdf["pred_mean"].values - bdf["obs"].values)))
    return {"ece": ece, "bins": bdf.to_dict(orient="records")}

print("\n=== Pooled ECE per horizon ===")
ece_table = {}
for name in ("RSF", "DeepSurv", "Ensemble"):
    print(f"\n  {name}:")
    ece_table[name] = {}
    for hi, h in enumerate(HORIZONS):
        res = pooled_ece(name, hi)
        if res is None:
            print(f"    {int(h):>3} mo:  insufficient data"); continue
        ece_table[name][int(h)] = res
        print(f"    {int(h):>3} mo:  ECE = {res['ece']:.4f}  ({len(res['bins'])} bins)")

# Save calibration bin tables for later plotting
for name in ("RSF", "DeepSurv", "Ensemble"):
    rows = []
    for hi, h in enumerate(HORIZONS):
        if int(h) not in ece_table[name]: continue
        for r in ece_table[name][int(h)]["bins"]:
            rows.append({**r, "horizon": int(h)})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f"calibration_bins_{name}.csv", index=False)


# --------------------------------------------------------------------------
# Classification at 36-month horizon
# --------------------------------------------------------------------------
print("\n=== Classification metrics at 36-month horizon ===")
horizon = 36
y_label = (events_all & (times_all < horizon)).astype(int)
mask_eval = (events_all == 1) | (times_all >= horizon)
print(f"  ({int(y_label[mask_eval].sum())} positives, "
      f"{int((~y_label[mask_eval].astype(bool)).sum())} negatives)\n")
print(f"  {'Model':9s}  AUC    Prec@33  Recall@33  F1@33   1-S(36) min..max")
print("  " + "-" * 65)
clf_records = []
for name in ("RSF", "DeepSurv", "Ensemble"):
    risk = OOF[name]["risk"]
    use = mask_eval & ~np.isnan(risk)
    r = risk[use]; lab = y_label[use]
    auc = roc_auc_score(lab, r)
    cutoff = np.quantile(r, 0.67)
    pred = (r >= cutoff).astype(int)
    p = precision_score(lab, pred, zero_division=0)
    rc = recall_score(lab, pred, zero_division=0)
    f = f1_score(lab, pred, zero_division=0)
    # also: spread of predicted ASD probability at 36mo
    s36 = OOF[name]["S"][:, HORIZONS.index(36.0)]
    s36 = s36[~np.isnan(s36)]
    p_event = 1 - s36
    clf_records.append({
        "model": name, "auc36": auc, "prec_top33": p, "recall_top33": rc, "f1_top33": f,
        "pevent36_min": float(p_event.min()), "pevent36_max": float(p_event.max()),
        "pevent36_spread": float(p_event.max() - p_event.min()),
    })
    print(f"  {name:9s}  {auc:.3f}  {p:.3f}    {rc:.3f}     {f:.3f}   "
          f"{p_event.min():.2f}..{p_event.max():.2f}  (spread {p_event.max()-p_event.min():.2f})")


# --------------------------------------------------------------------------
# Save everything
# --------------------------------------------------------------------------
out = {
    "oof_cindex": oof_cindex,
    "stability": stability_summary,
    "ece": {name: {int(h): ece_table[name][int(h)]["ece"] for h in HORIZONS if int(h) in ece_table[name]}
            for name in ece_table},
    "classification_36mo": clf_records,
}
with open(OUT / "calibration_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved -> {OUT / 'calibration_results.json'}")
print(f"Saved -> {OUT / 'fold_metrics.csv'}")
