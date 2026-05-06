"""
Sensitivity sweep: re-run honest 10-fold RSF CV under four cohort definitions.

  1. Primary             : full cohort (n=546)
  2. No pre-existing ASD : exclude ASD B4 Surgery == 1
  3. >=12 mo follow-up   : keep all events + censored with time_surv >= 12 mo
                           (the dataset dictionary's stated minimum)
  4. Both                : both filters applied

Reports cohort size, event count, C-index, IBS, and time-dep AUC at
12/24/36/48/60 mo for each variant. Also reports per-fold variance to
catch the "more stable" claim.

Intent: see whether stricter cohort definitions yield a meaningfully
better C-index, and whether the manuscript should consider switching
the primary cohort definition.
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
HORIZONS = np.array([12.0, 24.0, 36.0, 48.0, 60.0])

# -----------------------------------------------------------------------------
# Load full cohort once
# -----------------------------------------------------------------------------
df_full = pd.read_csv(ROOT / "cleaned_data.csv")
df_full.columns = df_full.columns.str.strip().str.replace("\n", " ", regex=False)
event_col = "REVERIFIED ASD"
df_full[event_col] = df_full[event_col].fillna(0).astype(int)
df_full["time_surv"] = np.where(
    df_full[event_col] == 1,
    df_full["Time Until ASD Diagnosis (months)"],
    df_full["Time Without_ASD (months)"],
)
df_full = df_full.dropna(subset=["time_surv"]).reset_index(drop=True)
df_full["time_surv"] = df_full["time_surv"].astype(float)


def filter_cohort(mode: str) -> pd.DataFrame:
    df = df_full.copy()
    if mode == "primary":
        return df.reset_index(drop=True)
    if mode == "no_preasd":
        return df[df["ASD B4 Surgery"] != 1].reset_index(drop=True)
    if mode == "ge12mo":
        # keep events at any time + censored with >=12 mo follow-up
        keep = (df[event_col] == 1) | (df["time_surv"] >= 12)
        return df[keep].reset_index(drop=True)
    if mode == "both":
        keep = ((df[event_col] == 1) | (df["time_surv"] >= 12)) & (df["ASD B4 Surgery"] != 1)
        return df[keep].reset_index(drop=True)
    raise ValueError(mode)


# -----------------------------------------------------------------------------
# Pipeline (honest, refit per fold) — same as honest_cv.py
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
numeric_like_cols = [
    "BMI", "ALIF Count", "Lateral Count",
    "Average PI", "PI-LL angle mismatch", "ABS PI-LL angle mismatch",
    "Post-op SS", "post PI", "post PT", "post LL", "post SVA",
    "length of hospital stay (d)", "levels_fused_count",
]


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


def rsf_factory():
    return RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )


def evaluate(df: pd.DataFrame, label: str) -> dict:
    present = [c for c in clean_feature_cols if c in df.columns]
    X_raw = df[present].copy()
    for c in numeric_like_cols:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
    str_cats = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=str_cats, drop_first=True).fillna(0.0)
    times = df["time_surv"].astype(float).values
    events = df[event_col].astype(bool).values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis, ibss = [], []
    auc_per_horizon = {h: [] for h in HORIZONS}
    ibs_grid = np.percentile(times, np.linspace(10, 90, 9))

    for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
        if events[te].sum() == 0:
            cis.append(np.nan); ibss.append(np.nan)
            for h in HORIZONS: auc_per_horizon[h].append(np.nan)
            continue
        try:
            transform = fit_pipeline(X_enc.iloc[tr], numeric_like_cols)
            X_tr = transform(X_enc.iloc[tr]); X_te = transform(X_enc.iloc[te])
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
                ibss.append(integrated_brier_score(y_tr, y_te, S, times=ibs_grid))
            except Exception:
                ibss.append(np.nan)

            ev_te = events[te]
            if ev_te.sum() >= 2:
                t_ev = times[te][ev_te]
                valid_h = HORIZONS[(HORIZONS > t_ev.min()) & (HORIZONS < t_ev.max())]
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
            else:
                for h in HORIZONS: auc_per_horizon[h].append(np.nan)
        except Exception as ex:
            print(f"  [{label}] fold {fold} FAILED: {ex}")
            cis.append(np.nan); ibss.append(np.nan)
            for h in HORIZONS: auc_per_horizon[h].append(np.nan)

    cis = np.array(cis, dtype=float); ibss = np.array(ibss, dtype=float)
    return {
        "label": label,
        "n": len(df),
        "n_events": int(events.sum()),
        "event_rate": float(events.sum() / len(df)),
        "cindex_mean": float(np.nanmean(cis)),
        "cindex_std": float(np.nanstd(cis)),
        "cindex_min": float(np.nanmin(cis)) if np.any(~np.isnan(cis)) else float("nan"),
        "cindex_max": float(np.nanmax(cis)) if np.any(~np.isnan(cis)) else float("nan"),
        "ibs_mean": float(np.nanmean(ibss)),
        "ibs_std": float(np.nanstd(ibss)),
        "auc_per_horizon": {
            int(h): {
                "mean": float(np.nanmean(auc_per_horizon[h])) if np.any(~np.isnan(auc_per_horizon[h])) else float("nan"),
                "std": float(np.nanstd(auc_per_horizon[h])) if np.any(~np.isnan(auc_per_horizon[h])) else float("nan"),
            } for h in HORIZONS
        },
    }


# -----------------------------------------------------------------------------
# Run all four variants
# -----------------------------------------------------------------------------
results = []
for mode, label in [
    ("primary",   "Primary (full)"),
    ("no_preasd", "No pre-existing ASD"),
    ("ge12mo",    ">=12 mo follow-up"),
    ("both",      "Both filters"),
]:
    df_sub = filter_cohort(mode)
    print(f"\n=== {label} (n={len(df_sub)}, events={int(df_sub[event_col].sum())}) ===")
    r = evaluate(df_sub, label)
    print(f"  C = {r['cindex_mean']:.3f} +/- {r['cindex_std']:.3f}  "
          f"(fold range {r['cindex_min']:.3f}-{r['cindex_max']:.3f})")
    print(f"  IBS = {r['ibs_mean']:.3f}")
    for h in HORIZONS:
        a = r['auc_per_horizon'][int(h)]['mean']
        print(f"  AUC@{int(h)}mo = {a:.3f}")
    results.append(r)

# -----------------------------------------------------------------------------
# Side-by-side summary table
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'Cohort':<24} {'n':>6} {'events':>8} {'C-index':>16} {'fold range':>14} "
      f"{'AUC@12':>8} {'AUC@24':>8} {'AUC@60':>8}")
print("=" * 100)
for r in results:
    print(f"{r['label']:<24} {r['n']:>6d} {r['n_events']:>8d} "
          f"{r['cindex_mean']:>8.3f} +/- {r['cindex_std']:>4.3f}   "
          f"{r['cindex_min']:>5.2f}-{r['cindex_max']:<5.2f}   "
          f"{r['auc_per_horizon'][12]['mean']:>6.3f}  "
          f"{r['auc_per_horizon'][24]['mean']:>6.3f}  "
          f"{r['auc_per_horizon'][60]['mean']:>6.3f}")
print("=" * 100)

with open(EXP / "sensitivity_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {EXP / 'sensitivity_sweep.json'}")
