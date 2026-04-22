"""
Try to beat the production Ensemble by improving RSF on its own merits:

  Variant A — Baseline RSF (matches current production rsf_bundle).
  Variant B — Baseline RSF + engineered clinical features.
  Variant C — Engineered features + deeper RSF (leaf=8).
  Variant D — Engineered features + wider feature subsetting (max_features=0.5).

For each variant: 10-fold honest CV, per-fold C-index / IBS / per-horizon AUC,
pooled OOF predictions at 12/24/36/60 mo, pooled ECE per horizon.

Then take the best variant, fit per-horizon isotonic calibration on its OOF
predictions, and re-evaluate ECE.

Finally: score the 8 clinical presets on the best calibrated variant and
compare against the production Ensemble + DeepSurv.

Output: experiments/rsf_improved_results.json
"""

from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Load + label
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
print(f"Cohort: {len(df)} patients, {int(df['REVERIFIED ASD'].sum())} events\n")

times_all = df["time_surv"].astype(float).values
events_all = df["REVERIFIED ASD"].astype(bool).values
N = len(df)

# --------------------------------------------------------------------------
# Base feature set (matches v1.4 production)
# --------------------------------------------------------------------------
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


def build_X(use_engineered: bool) -> pd.DataFrame:
    """Construct the feature matrix. If use_engineered, add clinically-
    motivated derived features on top of the base set."""
    X = df[present].copy()
    for c in numeric_like:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if use_engineered:
        # Pre-coerce numerics we need for engineering
        avg_pi = pd.to_numeric(X.get("Average PI"), errors="coerce").fillna(0)
        abs_mm = pd.to_numeric(X.get("ABS PI-LL angle mismatch"), errors="coerce").fillna(0)
        post_pt = pd.to_numeric(X.get("post PT"), errors="coerce").fillna(0)
        post_sva = pd.to_numeric(X.get("post SVA"), errors="coerce").fillna(0)
        age = pd.to_numeric(X.get("Age"), errors="coerce").fillna(0)
        prior = pd.to_numeric(X.get("prior back surgeries? (y=1)"), errors="coerce").fillna(0).astype(int)
        osteo = pd.to_numeric(X.get("Osteotomies (yes/no)"), errors="coerce").fillna(0).astype(int)
        ap = pd.to_numeric(X.get("Anterior + Posterior Apporoach"), errors="coerce").fillna(0).astype(int)
        defm_text = (
            (pd.to_numeric(X.get("dx_flat_back"), errors="coerce").fillna(0).astype(int)) |
            (pd.to_numeric(X.get("dx_scoliosis"), errors="coerce").fillna(0).astype(int)) |
            (pd.to_numeric(X.get("dx_sagittal_imbalance"), errors="coerce").fillna(0).astype(int))
        )

        levels = sum(
            pd.to_numeric(X.get(c), errors="coerce").fillna(0).astype(int)
            for c in ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
        )
        lumbosacral_span = (
            pd.to_numeric(X.get("L4-L5"), errors="coerce").fillna(0).astype(int) |
            pd.to_numeric(X.get("L5-S1"), errors="coerce").fillna(0).astype(int)
        )

        # Clinically-motivated engineered features:
        X["eng_levels_fused_count"] = levels.astype(float)
        X["eng_long_construct"] = (levels >= 4).astype(int)
        X["eng_short_construct"] = (levels <= 1).astype(int)
        X["eng_lumbosacral_span"] = lumbosacral_span.astype(int)
        X["eng_deformity_any"] = defm_text.astype(int)
        X["eng_revision_x_multilevel"] = (prior * (levels >= 3).astype(int)).astype(int)
        X["eng_osteo_x_long"] = (osteo * (levels >= 4).astype(int)).astype(int)
        X["eng_pill_x_age70"] = (((abs_mm >= 15).astype(int)) * ((age >= 70).astype(int))).astype(int)
        X["eng_pill_x_levels"] = (abs_mm * levels).astype(float)
        X["eng_sva_x_age"] = (post_sva * age).astype(float)
        X["eng_pt_x_pi"] = (post_pt * avg_pi).astype(float)
        X["eng_ap_x_long"] = (ap * (levels >= 4).astype(int)).astype(int)

    str_cats = X.select_dtypes(include=["object"]).columns.tolist()
    X_enc = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)
    return X_enc


# --------------------------------------------------------------------------
# Per-fold preprocessing pipeline
# --------------------------------------------------------------------------
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


HORIZONS = [12.0, 24.0, 36.0, 48.0, 60.0]
N_SPLITS = 10


def rsf_factory(leaf: int = 12, max_features="sqrt"):
    def _f():
        return RandomSurvivalForest(
            n_estimators=200, min_samples_split=4, min_samples_leaf=leaf,
            max_features=max_features, n_jobs=-1, random_state=42,
        )
    return _f


def surv_at(model, X, horizons):
    funcs = model.predict_survival_function(X, return_array=False)
    out = np.zeros((len(funcs), len(horizons)))
    for i, fn in enumerate(funcs):
        out[i] = np.interp(horizons, fn.x, fn.y, left=1.0, right=fn.y[-1])
    return out


# --------------------------------------------------------------------------
# CV runner — returns per-fold metrics + OOF risk + OOF survival at HORIZONS
# --------------------------------------------------------------------------
def run_cv(name: str, use_engineered: bool, factory):
    print(f"\n=== {name} (engineered={use_engineered}) ===")
    X_enc = build_X(use_engineered)
    print(f"  feature matrix: {X_enc.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    OOF_risk = np.full(N, np.nan)
    OOF_S = np.full((N, len(HORIZONS)), np.nan)
    fold_rows = []

    numeric_present_list = numeric_like + [
        "eng_levels_fused_count","eng_pill_x_levels","eng_sva_x_age","eng_pt_x_pi",
    ] if use_engineered else numeric_like

    for fold, (tr, te) in enumerate(skf.split(X_enc.values, events_all.astype(int)), 1):
        if events_all[te].sum() == 0:
            continue
        transform = fit_pipeline(X_enc.iloc[tr], numeric_present_list)
        Xtr, Xte = transform(X_enc.iloc[tr]), transform(X_enc.iloc[te])
        ytr = Surv.from_arrays(event=events_all[tr], time=times_all[tr])
        yte = Surv.from_arrays(event=events_all[te], time=times_all[te])

        m = factory()
        m.fit(Xtr, ytr)
        risk = m.predict(Xte)
        OOF_risk[te] = risk
        OOF_S[te] = surv_at(m, Xte, HORIZONS)

        ci = concordance_index_censored(events_all[te], times_all[te], risk)[0]

        # IBS on test-fold time grid
        t_max = min(times_all[tr].max(), times_all[te].max()) - 1e-3
        t_min = max(times_all[tr].min(), times_all[te].min()) + 1e-3
        ibs_grid = np.linspace(t_min, t_max, 10)
        try:
            S_ibs = surv_at(m, Xte, list(ibs_grid))
            ibs = integrated_brier_score(ytr, yte, S_ibs, times=ibs_grid)
        except Exception:
            ibs = float("nan")

        # Per-horizon AUC
        auc_dict = {}
        valid_h = [h for h in HORIZONS if t_min < h < t_max]
        if valid_h:
            try:
                aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, np.asarray(valid_h, dtype=float))
                aucs = np.atleast_1d(aucs)
                auc_dict = {f"auc_{int(h)}": float(a) for h, a in zip(valid_h, aucs)}
            except Exception:
                pass

        fold_rows.append({"fold": fold, "cindex": ci, "ibs": ibs, **auc_dict})

    fdf = pd.DataFrame(fold_rows)
    print(f"  C-index = {fdf['cindex'].mean():.3f} +/- {fdf['cindex'].std():.3f}  "
          f"[{fdf['cindex'].min():.3f}, {fdf['cindex'].max():.3f}]")
    print(f"  IBS     = {fdf['ibs'].mean():.3f} +/- {fdf['ibs'].std():.3f}")
    if "auc_24" in fdf:
        print(f"  AUC@24  = {fdf['auc_24'].mean():.3f} +/- {fdf['auc_24'].std():.3f}")
    if "auc_60" in fdf:
        print(f"  AUC@60  = {fdf['auc_60'].mean():.3f} +/- {fdf['auc_60'].std():.3f}")

    # Pooled OOF C-index
    valid = ~np.isnan(OOF_risk)
    oof_ci = concordance_index_censored(events_all[valid], times_all[valid], OOF_risk[valid])[0]
    print(f"  OOF pooled C = {oof_ci:.3f}")

    return {
        "name": name,
        "engineered": use_engineered,
        "fold_metrics": fdf,
        "oof_risk": OOF_risk,
        "oof_S": OOF_S,
        "oof_cindex": float(oof_ci),
        "X_enc": X_enc,
    }


# --------------------------------------------------------------------------
# Pooled ECE per horizon (KM-based observed in each prediction bin)
# --------------------------------------------------------------------------
def pooled_ece(name: str, OOF_S: np.ndarray, n_bins: int = 10):
    out = {}
    for hi, h in enumerate(HORIZONS):
        S = OOF_S[:, hi]
        valid = ~np.isnan(S)
        preds = S[valid]
        t = times_all[valid]; e = events_all[valid]
        if len(preds) < n_bins * 2:
            continue
        try:
            bin_id = pd.qcut(pd.Series(preds).rank(method="first"),
                             q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for b in sorted(np.unique(bin_id)):
            mask = bin_id == b
            if mask.sum() < 3: continue
            try:
                km = KaplanMeierFitter().fit(t[mask], event_observed=e[mask])
                obs = float(km.predict(h))
            except Exception:
                obs = float("nan")
            rows.append({"n": int(mask.sum()),
                         "pred_mean": float(np.mean(preds[mask])), "obs": obs})
        if not rows: continue
        bdf = pd.DataFrame(rows).dropna(subset=["obs"])
        c = bdf["n"].values.astype(float)
        out[int(h)] = float(np.sum((c / c.sum()) * np.abs(bdf["pred_mean"].values - bdf["obs"].values)))
    return out


# --------------------------------------------------------------------------
# Per-horizon isotonic calibration of survival probabilities
# --------------------------------------------------------------------------
def fit_isotonic_calibrators(OOF_S: np.ndarray) -> dict[int, IsotonicRegression]:
    """For each horizon h: take patients evaluable at h (event by h, or
    censored/event-free past h), fit an isotonic regression mapping
    predicted P(event by h) -> observed event indicator. Apply at serving
    time to convert raw 1-S(h) into calibrated probability."""
    out = {}
    for hi, h in enumerate(HORIZONS):
        S = OOF_S[:, hi]
        valid = ~np.isnan(S)
        # evaluable: had the event before h, OR survived past h
        pre_event = (events_all & (times_all < h))
        survived = (times_all >= h)
        evaluable = (pre_event | survived) & valid
        if evaluable.sum() < 30:
            continue
        p_event_pred = (1.0 - S[evaluable])
        y_event = (events_all[evaluable] & (times_all[evaluable] < h)).astype(int)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p_event_pred, y_event)
        out[int(h)] = ir
    return out


def apply_calibration(OOF_S: np.ndarray, calibrators: dict[int, IsotonicRegression]) -> np.ndarray:
    out = OOF_S.copy()
    for hi, h in enumerate(HORIZONS):
        ir = calibrators.get(int(h))
        if ir is None: continue
        S = out[:, hi]
        valid = ~np.isnan(S)
        p_event_cal = ir.transform(1.0 - S[valid])
        out[valid, hi] = 1.0 - p_event_cal
    return out


# --------------------------------------------------------------------------
# Run all variants
# --------------------------------------------------------------------------
results = {}

results["A_baseline"] = run_cv("A. Baseline RSF (current production)",
                                use_engineered=False,
                                factory=rsf_factory(leaf=12, max_features="sqrt"))

results["B_engineered"] = run_cv("B. Engineered features + baseline RSF",
                                  use_engineered=True,
                                  factory=rsf_factory(leaf=12, max_features="sqrt"))

results["C_eng_deeper"] = run_cv("C. Engineered features + deeper RSF (leaf=8)",
                                  use_engineered=True,
                                  factory=rsf_factory(leaf=8, max_features="sqrt"))

results["D_eng_wider"] = run_cv("D. Engineered features + wider features (max_features=0.5)",
                                 use_engineered=True,
                                 factory=rsf_factory(leaf=12, max_features=0.5))

# --------------------------------------------------------------------------
# Per-horizon ECE for each variant
# --------------------------------------------------------------------------
print("\n=== Pooled ECE per horizon (uncalibrated) ===")
print(f"  {'Variant':22s}  {'12mo':>6s}  {'24mo':>6s}  {'36mo':>6s}  {'48mo':>6s}  {'60mo':>6s}")
for k, r in results.items():
    ece = pooled_ece(r["name"], r["oof_S"])
    r["ece"] = ece
    print(f"  {k:22s}  " + "  ".join(f"{ece.get(int(h), float('nan')):.4f}" for h in HORIZONS))

# --------------------------------------------------------------------------
# Pick the winner: best OOF C-index AND best (mean) ECE across horizons
# --------------------------------------------------------------------------
def score_variant(r):
    cidx = r["oof_cindex"]
    ece_vals = [v for v in r["ece"].values()]
    ece_mean = float(np.mean(ece_vals)) if ece_vals else float("nan")
    return {"oof_cindex": cidx, "ece_mean": ece_mean}

print("\n=== Variant scoring ===")
for k, r in results.items():
    s = score_variant(r)
    print(f"  {k:22s}  C={s['oof_cindex']:.3f}  mean ECE={s['ece_mean']:.4f}")

# Winner = highest C-index minus mean-ECE penalty
def composite(r):
    s = score_variant(r)
    return s["oof_cindex"] - s["ece_mean"]   # higher is better

winner_key = max(results.keys(), key=lambda k: composite(results[k]))
print(f"\nWinner pre-calibration: {winner_key}")
winner = results[winner_key]

# --------------------------------------------------------------------------
# Apply isotonic calibration to the winner
# --------------------------------------------------------------------------
print("\n=== Isotonic calibration of winner ===")
calibrators = fit_isotonic_calibrators(winner["oof_S"])
print(f"  Fit calibrators for {sorted(calibrators.keys())} mo horizons")
S_cal = apply_calibration(winner["oof_S"], calibrators)
ece_cal = pooled_ece(winner["name"] + " (calibrated)", S_cal)
print(f"  ECE before:  " + "  ".join(f"{winner['ece'].get(int(h), float('nan')):.4f}" for h in HORIZONS))
print(f"  ECE after :  " + "  ".join(f"{ece_cal.get(int(h), float('nan')):.4f}" for h in HORIZONS))

# --------------------------------------------------------------------------
# 8-preset clinical sanity check on the calibrated winner (full-data fit)
# --------------------------------------------------------------------------
PRESETS = {
    "Low risk short": dict(
        Sex=1, Age=50, BMI=24, prior_back_surgeries=0,
        dx_stenosis=1, L4_L5=1, L5_S1=0,
        Open=0, Perc_screws=1,
        Standalone_XLIF_Check=1, Retroperitoneal_Approach=1,
        Lateral_Count=0, ALIF_Count=0,
        Average_PI=48, PI_LL=4, ABS_PI_LL=4, PI_gt_50=0,
        post_LL=50, post_SVA=10, post_PI=48, post_PT=13, Post_op_SS=35, LOS=2,
    ),
    "Adjacent stress": dict(
        Sex=1, Age=63, BMI=28, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=0, Lateral_Count=1, ALIF_Count=0,
        Average_PI=58, PI_LL=15, ABS_PI_LL=15, PI_gt_50=1,
        post_LL=45, post_SVA=35, post_PI=58, post_PT=22, Post_op_SS=28, LOS=4,
    ),
    "Deformity / long construct": dict(
        Sex=1, Age=75, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_scoliosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1, AP_approach=1, ACR=1,
        Lateral_Count=3, ALIF_Count=0,
        Average_PI=65, PI_LL=25, ABS_PI_LL=25, PI_gt_50=1,
        post_LL=40, post_SVA=65, post_PI=65, post_PT=30, Post_op_SS=20, LOS=8,
    ),
    "Single-level spondy": dict(
        Sex=1, Age=60, BMI=27, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylolisthesis=1,
        L4_L5=1, L5_S1=0,
        Open=1, Lateral_Count=1, ALIF_Count=0,
        Average_PI=52, PI_LL=10, ABS_PI_LL=10, PI_gt_50=1,
        post_LL=45, post_SVA=25, post_PI=52, post_PT=16, Post_op_SS=30, LOS=3,
    ),
    "Revision lumbar": dict(
        Sex=1, Age=68, BMI=30, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=1, Osteotomies=1, AP_approach=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=56, PI_LL=18, ABS_PI_LL=18, PI_gt_50=1,
        post_LL=45, post_SVA=40, post_PI=56, post_PT=24, Post_op_SS=27, LOS=5,
    ),
    "Flat-back correction": dict(
        Sex=1, Age=71, BMI=29, prior_back_surgeries=1,
        dx_stenosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1, AP_approach=1, ACR=1,
        Lateral_Count=2, ALIF_Count=0,
        Average_PI=60, PI_LL=25, ABS_PI_LL=25, PI_gt_50=1,
        post_LL=42, post_SVA=55, post_PI=60, post_PT=28, Post_op_SS=22, LOS=8,
    ),
    "MIS short fusion": dict(
        Sex=1, Age=55, BMI=26, prior_back_surgeries=0,
        dx_stenosis=1, L4_L5=1,
        Open=0, Perc_screws=1,
        Standalone_XLIF_Check=1, Retroperitoneal_Approach=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=50, PI_LL=6, ABS_PI_LL=6, PI_gt_50=1,
        post_LL=48, post_SVA=18, post_PI=50, post_PT=14, Post_op_SS=33, LOS=2,
    ),
    "Elderly mismatch": dict(
        Sex=1, Age=78, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1, dx_sagittal_imbalance=1,
        L4_L5=1, L5_S1=1,
        Open=1, Osteotomies=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=62, PI_LL=22, ABS_PI_LL=22, PI_gt_50=1,
        post_LL=45, post_SVA=50, post_PI=62, post_PT=26, Post_op_SS=25, LOS=6,
    ),
}

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

# Fit winner on full data
winner_X = winner["X_enc"]
print(f"\n=== Fitting winner ({winner_key}) on full cohort ===")

# Compute the engineered features for presets the same way build_X did
def preset_to_row(preset: dict, template_cols: list[str], use_eng: bool) -> pd.DataFrame:
    row = {c: 0 for c in template_cols}
    for k, v in preset.items():
        col = KEY_TO_COL.get(k)
        if col and col in row:
            row[col] = v
    rdf = pd.DataFrame([row])
    if use_eng:
        # Recompute engineered features (must match build_X exactly)
        levels = sum(int(rdf.iloc[0].get(c, 0)) for c in
                     ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"])
        lumbosac = int(rdf.iloc[0].get("L4-L5", 0)) | int(rdf.iloc[0].get("L5-S1", 0))
        defm_any = int(rdf.iloc[0].get("dx_flat_back", 0)) | int(rdf.iloc[0].get("dx_scoliosis", 0)) | int(rdf.iloc[0].get("dx_sagittal_imbalance", 0))
        prior = int(rdf.iloc[0].get("prior back surgeries? (y=1)", 0))
        osteo = int(rdf.iloc[0].get("Osteotomies (yes/no)", 0))
        ap = int(rdf.iloc[0].get("Anterior + Posterior Apporoach", 0))
        avg_pi = float(rdf.iloc[0].get("Average PI", 0) or 0)
        abs_mm = float(rdf.iloc[0].get("ABS PI-LL angle mismatch", 0) or 0)
        post_pt = float(rdf.iloc[0].get("post PT", 0) or 0)
        post_sva = float(rdf.iloc[0].get("post SVA", 0) or 0)
        age = float(rdf.iloc[0].get("Age", 0) or 0)

        rdf["eng_levels_fused_count"] = float(levels)
        rdf["eng_long_construct"] = int(levels >= 4)
        rdf["eng_short_construct"] = int(levels <= 1)
        rdf["eng_lumbosacral_span"] = lumbosac
        rdf["eng_deformity_any"] = defm_any
        rdf["eng_revision_x_multilevel"] = prior * int(levels >= 3)
        rdf["eng_osteo_x_long"] = osteo * int(levels >= 4)
        rdf["eng_pill_x_age70"] = int(abs_mm >= 15) * int(age >= 70)
        rdf["eng_pill_x_levels"] = abs_mm * levels
        rdf["eng_sva_x_age"] = post_sva * age
        rdf["eng_pt_x_pi"] = post_pt * avg_pi
        rdf["eng_ap_x_long"] = ap * int(levels >= 4)

    # Reindex to template cols (won't include eng_* if they're not in template)
    return rdf.reindex(columns=template_cols, fill_value=0)


winner_uses_eng = winner["engineered"]
template_cols = winner["X_enc"].columns.tolist()
numeric_present_list = numeric_like + (
    ["eng_levels_fused_count","eng_pill_x_levels","eng_sva_x_age","eng_pt_x_pi"]
    if winner_uses_eng else []
)
# Final-fit pipeline on the winner's full feature matrix
final_transform = fit_pipeline(winner_X, numeric_present_list)
X_full = final_transform(winner_X)
y_full = Surv.from_arrays(event=events_all, time=times_all)
final_factory = (
    rsf_factory(leaf=12, max_features="sqrt") if winner_key in ("A_baseline", "B_engineered")
    else rsf_factory(leaf=8, max_features="sqrt") if winner_key == "C_eng_deeper"
    else rsf_factory(leaf=12, max_features=0.5)
)
final_rsf = final_factory()
final_rsf.fit(X_full, y_full)
cohort_risk = final_rsf.predict(X_full)

print("\n=== 8-preset clinical sanity check on calibrated winner ===")
print(f"  {'Preset':30s}  pct   1y     2y     3y     5y    median")
print("  " + "-" * 70)
preset_records = []
for pname, pdict in PRESETS.items():
    raw_row = preset_to_row(pdict, template_cols, winner_uses_eng)
    Xp = final_transform(raw_row)
    risk = float(final_rsf.predict(Xp)[0])
    pct = float((cohort_risk < risk).mean() * 100)
    S_raw = surv_at(final_rsf, Xp, HORIZONS).reshape(-1)
    # Apply calibration
    S_cal_p = S_raw.copy()
    for hi, h in enumerate(HORIZONS):
        ir = calibrators.get(int(h))
        if ir is None: continue
        p_ev = float(ir.transform(np.array([1.0 - S_raw[hi]]))[0])
        S_cal_p[hi] = 1.0 - p_ev

    horizons_dict = {int(h): float(s) for h, s in zip(HORIZONS, S_cal_p)}
    median = None
    surv_fn = final_rsf.predict_survival_function(Xp, return_array=False)[0]
    for t, s in zip(surv_fn.x, surv_fn.y):
        if s <= 0.5: median = float(t); break
    med_str = f"{median:.0f}mo" if median else "  >FU"
    print(f"  {pname:30s}  {pct:>4.0f}  "
          f"{horizons_dict[12]*100:5.1f}  {horizons_dict[24]*100:5.1f}  "
          f"{horizons_dict[36]*100:5.1f}  {horizons_dict[60]*100:5.1f}  {med_str:>6s}")
    preset_records.append({
        "preset": pname, "pct": pct,
        "p1y": horizons_dict[12], "p2y": horizons_dict[24],
        "p3y": horizons_dict[36], "p5y": horizons_dict[60],
        "median": median,
    })

# Spread
spread = max(r["p5y"] for r in preset_records) - min(r["p5y"] for r in preset_records)
print(f"\n  5y range: {min(r['p5y'] for r in preset_records)*100:.1f}%-"
      f"{max(r['p5y'] for r in preset_records)*100:.1f}%  (spread {spread*100:.1f} pts)")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
summary = {
    "variants": {
        k: {
            "name": v["name"], "engineered": v["engineered"],
            "oof_cindex": v["oof_cindex"],
            "fold_cindex_mean": float(v["fold_metrics"]["cindex"].mean()),
            "fold_cindex_std": float(v["fold_metrics"]["cindex"].std()),
            "fold_cindex_min": float(v["fold_metrics"]["cindex"].min()),
            "fold_cindex_max": float(v["fold_metrics"]["cindex"].max()),
            "fold_ibs_mean": float(v["fold_metrics"]["ibs"].mean()),
            "ece": v["ece"],
        }
        for k, v in results.items()
    },
    "winner": winner_key,
    "winner_calibrated_ece": ece_cal,
    "preset_predictions": preset_records,
    "preset_5y_spread_pts": float(spread * 100),
}
with open(OUT / "rsf_improved_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\nSaved -> {OUT / 'rsf_improved_results.json'}")
