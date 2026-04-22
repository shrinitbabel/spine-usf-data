"""
Extended model exploration:

  1. Time-dependent AUC for the production RSF model (clinical record).
  2. Inter-model risk-score correlation — which pairs are complementary?
  3. Rank-averaging ensembles (every pair + the full set) — does ensembling help?
  4. ComponentwiseGradientBoosting.
  5. DeepSurv (small MLP, dropout regularised — small cohort, no overfit room).
  6. Horizon-specific precision/recall at the 36-month cutoff.

All evaluation is honest CV (PCA + scaler + rare-col drop refit per fold).
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

from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
    ComponentwiseGradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Load (matches honest_cv.py exactly)
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
times = df["time_surv"].astype(float).values
events = df["REVERIFIED ASD"].astype(bool).values

print(f"Cohort: {len(df)} patients, {int(events.sum())} events, {X_enc.shape[1]} encoded cols\n")


# --------------------------------------------------------------------------
# Per-fold preprocessing
# --------------------------------------------------------------------------
def fit_pipeline(X_tr: pd.DataFrame, numeric_present):
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
# Model factories
# --------------------------------------------------------------------------
def rsf():
    return RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )

def gbsa():
    return GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
    )

def coxnet():
    return CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=20, fit_baseline_model=True)

def cwgb():
    return ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, random_state=42,
    )


# --------------------------------------------------------------------------
# Honest CV: collect per-fold risk vectors so we can ensemble & correlate
# --------------------------------------------------------------------------
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# OOF risk score storage: each model fills a length-N vector with NaN where
# the patient was in the training fold. After the loop it's a complete OOF set.
N = len(df)
OOF = {name: np.full(N, np.nan) for name in ("RSF", "GBSA", "Coxnet", "CWGB", "DeepSurv")}

# Time-dependent AUC accumulator for RSF
auc_horizons = [12, 24, 36, 48, 60, 72]
rsf_auc_per_fold = []  # list of (auc_at_each_horizon)

# DeepSurv setup
DEEP_OK = True
try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH as CoxPH_NN
    torch.manual_seed(42)
except Exception as ex:
    print(f"DeepSurv unavailable ({ex})")
    DEEP_OK = False


def fit_deepsurv(X_tr, t_tr, e_tr, X_te):
    """Tiny MLP, heavy dropout — small cohort, no room to overfit."""
    in_features = X_tr.shape[1]
    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[32, 16],
        out_features=1,
        batch_norm=True,
        dropout=0.5,
        output_bias=False,
    )
    model = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
    y_tr = (t_tr.astype("float32"), e_tr.astype("float32"))
    model.fit(
        X_tr.astype("float32"), y_tr,
        batch_size=64, epochs=80, verbose=False,
    )
    model.compute_baseline_hazards()
    return model.predict(X_te.astype("float32")).reshape(-1)


for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
    if events[te].sum() == 0:
        continue
    transform = fit_pipeline(X_enc.iloc[tr], numeric_like)
    Xtr, Xte = transform(X_enc.iloc[tr]), transform(X_enc.iloc[te])
    ytr = Surv.from_arrays(event=events[tr], time=times[tr])
    yte = Surv.from_arrays(event=events[te], time=times[te])

    for name, factory in (("RSF", rsf), ("GBSA", gbsa), ("Coxnet", coxnet), ("CWGB", cwgb)):
        try:
            m = factory()
            m.fit(Xtr, ytr)
            OOF[name][te] = m.predict(Xte)
        except Exception as ex:
            print(f"  fold {fold} {name} failed: {ex}")

    if DEEP_OK:
        try:
            risk = fit_deepsurv(Xtr, times[tr], events[tr], Xte)
            OOF["DeepSurv"][te] = risk
        except Exception as ex:
            print(f"  fold {fold} DeepSurv failed: {ex}")

    # RSF time-AUC for this fold — must be inside the test-fold follow-up
    # window AND not exceed the training-fold max time, or sksurv refuses.
    try:
        m_rsf = rsf().fit(Xtr, ytr)
        risk_te = m_rsf.predict(Xte)
        t_max = min(times[tr].max(), times[te].max()) - 1e-3
        t_min = max(times[tr].min(), times[te].min()) + 1e-3
        valid_h = [h for h in auc_horizons if t_min < h < t_max]
        if valid_h:
            aucs, _ = cumulative_dynamic_auc(ytr, yte, risk_te, np.asarray(valid_h, dtype=float))
            aucs = np.atleast_1d(aucs)
            row = {h: float(a) for h, a in zip(valid_h, aucs)}
            rsf_auc_per_fold.append(row)
    except Exception as ex:
        print(f"  fold {fold} time-AUC failed: {ex}")

print("Folds done.\n")


# --------------------------------------------------------------------------
# 1. Per-model OOF C-index
# --------------------------------------------------------------------------
print("=== OOF C-index per model (single concordance over all folds) ===")
ci_per_model = {}
for name, oof in OOF.items():
    valid = ~np.isnan(oof)
    if valid.sum() == 0:
        print(f"  {name:9s}  no predictions"); continue
    ci = concordance_index_censored(events[valid], times[valid], oof[valid])[0]
    ci_per_model[name] = ci
    print(f"  {name:9s}  C = {ci:.3f}  (n={int(valid.sum())})")

# --------------------------------------------------------------------------
# 2. Inter-model correlation of risk scores (Spearman, on rank)
# --------------------------------------------------------------------------
print("\n=== Spearman rank-correlation of OOF risk scores ===")
names = [n for n, v in OOF.items() if not np.all(np.isnan(v))]
ranks = {n: pd.Series(OOF[n]).rank() for n in names}
corr = pd.DataFrame(
    {a: {b: ranks[a].corr(ranks[b]) for b in names} for a in names}
)
print(corr.round(3).to_string())
print("\n(Lower correlation = more independent signal = better ensemble candidate.)")

# --------------------------------------------------------------------------
# 3. Rank-averaging ensembles
# --------------------------------------------------------------------------
print("\n=== Ensemble C-index (rank-average of OOF risks) ===")
from itertools import combinations

def ensemble_ci(model_subset):
    valid = np.ones(N, dtype=bool)
    for m in model_subset:
        valid &= ~np.isnan(OOF[m])
    if valid.sum() == 0:
        return None
    rs = np.zeros(valid.sum())
    for m in model_subset:
        rs += pd.Series(OOF[m][valid]).rank().values
    rs /= len(model_subset)
    return concordance_index_censored(events[valid], times[valid], rs)[0]

ensemble_results = {}
# All pairs
for combo in combinations(names, 2):
    ci = ensemble_ci(combo)
    if ci is not None:
        key = "+".join(combo)
        ensemble_results[key] = ci
        print(f"  {key:30s}  C = {ci:.3f}")

# All triples
for combo in combinations(names, 3):
    ci = ensemble_ci(combo)
    if ci is not None:
        key = "+".join(combo)
        ensemble_results[key] = ci
        print(f"  {key:30s}  C = {ci:.3f}")

# Full set
full = tuple(names)
ci = ensemble_ci(full)
if ci is not None:
    key = "+".join(full)
    ensemble_results[key] = ci
    print(f"  {key:30s}  C = {ci:.3f}")

# --------------------------------------------------------------------------
# 4. Time-dependent AUC for RSF (clinical record)
# --------------------------------------------------------------------------
print("\n=== RSF time-dependent AUC (mean over folds, where defined) ===")
auc_rows = []
for h in auc_horizons:
    vals = [r[h] for r in rsf_auc_per_fold if h in r and not np.isnan(r[h])]
    if vals:
        m, s = float(np.mean(vals)), float(np.std(vals))
        print(f"  {h:>3} mo:  AUC = {m:.3f} +/- {s:.3f}  (n folds = {len(vals)})")
        auc_rows.append({"horizon_months": h, "auc_mean": m, "auc_std": s, "n_folds": len(vals)})
    else:
        print(f"  {h:>3} mo:  no valid folds")

# --------------------------------------------------------------------------
# 5. Precision / recall at 36-month cutoff (event before vs after)
# --------------------------------------------------------------------------
print("\n=== Classification metrics at 36-month horizon ===")
print("(Events with time < 36 mo and observed = 1 are 'positives'.)\n")
horizon = 36
y_label = (events & (times < horizon)).astype(int)
mask_valid = (events == 1) | (times >= horizon)  # exclude censored before horizon
print(f"  {y_label[mask_valid].sum()} positives, {(~y_label[mask_valid].astype(bool)).sum()} negatives "
      f"(of {mask_valid.sum()} evaluable patients)")

print("\n  Model      AUC   Prec@top33%   Recall@top33%   F1@top33%")
print("  " + "-" * 55)
clf_records = []
for name in names:
    oof = OOF[name]
    use = mask_valid & ~np.isnan(oof)
    if use.sum() < 10:
        continue
    risks = oof[use]
    labels = y_label[use]
    try:
        auc = roc_auc_score(labels, risks)
    except Exception:
        auc = float("nan")
    # Pick top 33% as predicted positives
    cutoff = np.quantile(risks, 0.67)
    pred = (risks >= cutoff).astype(int)
    p = precision_score(labels, pred, zero_division=0)
    r = recall_score(labels, pred, zero_division=0)
    f = f1_score(labels, pred, zero_division=0)
    print(f"  {name:9s}  {auc:.3f}    {p:.3f}        {r:.3f}          {f:.3f}")
    clf_records.append({"model": name, "auc36": auc, "prec_top33": p, "recall_top33": r, "f1_top33": f})

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
results = {
    "oof_cindex": ci_per_model,
    "spearman_corr": corr.to_dict(),
    "ensembles": ensemble_results,
    "rsf_time_auc": auc_rows,
    "horizon_36mo_classification": clf_records,
}
with open(OUT / "extended_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved -> {OUT / 'extended_results.json'}")
