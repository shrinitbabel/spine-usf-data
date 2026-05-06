"""
Multi-model sensitivity sweep: retrain RSF, DeepSurv, and GBSA under four
cohort definitions, with honest 10-fold CV in each combination. Uses the
same AUC bounding logic as extended_models.py so numbers reconcile with
the existing manuscript tables.

Cohorts:
  1. Primary               : full cohort (n=546)
  2. No pre-existing ASD   : ASD B4 Surgery != 1
  3. >=12 mo follow-up     : keep all events + censored with time_surv >= 12
  4. Both                  : both filters

Models:
  - RSF (deployed primary)
  - DeepSurv (deployed secondary)
  - GBSA   (third-strongest in original analysis)

Reports per (cohort, model): C-index (mean +/- std + fold range), IBS,
time-dep AUC at 12/24/36/48/60 mo. Also reports DeepSurv-RSF Spearman
risk correlation per cohort.

Saves: experiments/sensitivity_multi_model.json
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
HORIZONS = [12.0, 24.0, 36.0, 48.0, 60.0]

# DeepSurv stack (optional)
DEEP_OK = True
try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH as CoxPH_NN
    torch.manual_seed(42)
except Exception as ex:
    print(f"DeepSurv unavailable: {ex}")
    DEEP_OK = False

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

CLEAN_COLS = [
    "Sex","Age","BMI","prior back surgeries? (y=1)",
    "dx_adjacent_segment","dx_spondylolisthesis","dx_spondylosis","dx_stenosis",
    "dx_scoliosis","dx_flat_back","dx_sagittal_imbalance","dx_post_laminectomy","dx_deformity",
    "Case/Type of Surgery","Additional Procedures w/in surgery",
    "levels_fused_count",
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
NUMERIC_LIKE = [
    "BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
    "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
    "post SVA","length of hospital stay (d)","levels_fused_count",
]


def filter_cohort(mode: str) -> pd.DataFrame:
    df = df_full.copy()
    if mode == "primary":          return df.reset_index(drop=True)
    if mode == "no_preasd":        return df[df["ASD B4 Surgery"] != 1].reset_index(drop=True)
    if mode == "ge12mo":           return df[(df[event_col] == 1) | (df["time_surv"] >= 12)].reset_index(drop=True)
    if mode == "both":             return df[((df[event_col] == 1) | (df["time_surv"] >= 12)) & (df["ASD B4 Surgery"] != 1)].reset_index(drop=True)
    raise ValueError(mode)


def encode(df):
    present = [c for c in CLEAN_COLS if c in df.columns]
    X_raw = df[present].copy()
    for c in NUMERIC_LIKE:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
    str_cats = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=str_cats, drop_first=True).fillna(0.0)
    times = df["time_surv"].astype(float).values
    events = df[event_col].astype(bool).values
    return X_enc, times, events


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


def rsf_factory():
    return RandomSurvivalForest(
        n_estimators=200, min_samples_split=4, min_samples_leaf=12,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )

def gbsa_factory():
    return GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
    )


def fit_deepsurv(X_tr, t_tr, e_tr, X_te):
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
    model.fit(X_tr.astype("float32"), y_tr, batch_size=64, epochs=80, verbose=False)
    model.compute_baseline_hazards()
    return model.predict(X_te.astype("float32")).reshape(-1)


def auc_horizons_for_fold(times_tr, times_te):
    """Match extended_models.py logic — bound by max(time) in both train & test."""
    t_max = min(times_tr.max(), times_te.max()) - 1e-3
    t_min = max(times_tr.min(), times_te.min()) + 1e-3
    return [h for h in HORIZONS if t_min < h < t_max]


def evaluate_model(name, X_enc, times, events, n_splits=10):
    """Honest 10-fold CV. Returns C-index, IBS, time-dep AUC per horizon, OOF risk."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cis, ibss = [], []
    auc_per_h = {h: [] for h in HORIZONS}
    oof_risk = np.full(len(X_enc), np.nan)
    ibs_grid = np.percentile(times, np.linspace(10, 90, 9))

    for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
        if events[te].sum() == 0:
            cis.append(np.nan); ibss.append(np.nan)
            for h in HORIZONS: auc_per_h[h].append(np.nan)
            continue
        try:
            transform = fit_pipeline(X_enc.iloc[tr], NUMERIC_LIKE)
            Xtr = transform(X_enc.iloc[tr]); Xte = transform(X_enc.iloc[te])
            ytr = Surv.from_arrays(event=events[tr], time=times[tr])
            yte = Surv.from_arrays(event=events[te], time=times[te])

            if name == "RSF":
                m = rsf_factory(); m.fit(Xtr, ytr)
                risk = m.predict(Xte)
                surv = m.predict_survival_function(Xte, return_array=False)
                S = np.zeros((len(surv), len(ibs_grid)))
                for i, fn in enumerate(surv):
                    S[i] = np.interp(ibs_grid, fn.x, fn.y, left=1.0, right=fn.y[-1])
                try: ibs = integrated_brier_score(ytr, yte, S, times=ibs_grid)
                except Exception: ibs = np.nan
            elif name == "GBSA":
                m = gbsa_factory(); m.fit(Xtr, ytr)
                risk = m.predict(Xte)
                ibs = np.nan
            elif name == "DeepSurv":
                if not DEEP_OK:
                    cis.append(np.nan); ibss.append(np.nan)
                    for h in HORIZONS: auc_per_h[h].append(np.nan)
                    continue
                risk = fit_deepsurv(Xtr, times[tr], events[tr], Xte)
                ibs = np.nan
            else:
                raise ValueError(name)

            oof_risk[te] = risk
            ci = concordance_index_censored(events[te], times[te], risk)[0]
            cis.append(ci); ibss.append(ibs)

            valid_h = auc_horizons_for_fold(times[tr], times[te])
            if valid_h:
                try:
                    aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, np.asarray(valid_h, dtype=float))
                    aucs = np.atleast_1d(aucs)
                    for h, a in zip(valid_h, aucs):
                        auc_per_h[h].append(float(a))
                except Exception:
                    pass
            for h in HORIZONS:
                if h not in valid_h:
                    auc_per_h[h].append(np.nan)
        except Exception as ex:
            print(f"  [{name}] fold {fold} FAILED: {ex}")
            cis.append(np.nan); ibss.append(np.nan)
            for h in HORIZONS: auc_per_h[h].append(np.nan)

    cis = np.array(cis, dtype=float); ibss = np.array(ibss, dtype=float)
    return {
        "model": name,
        "cindex_mean": float(np.nanmean(cis)),
        "cindex_std":  float(np.nanstd(cis)),
        "cindex_min":  float(np.nanmin(cis)) if np.any(~np.isnan(cis)) else float("nan"),
        "cindex_max":  float(np.nanmax(cis)) if np.any(~np.isnan(cis)) else float("nan"),
        "ibs_mean":    float(np.nanmean(ibss)) if np.any(~np.isnan(ibss)) else float("nan"),
        "auc_per_horizon": {
            int(h): float(np.nanmean(auc_per_h[h])) if np.any(~np.isnan(auc_per_h[h])) else float("nan")
            for h in HORIZONS
        },
        "oof_risk": oof_risk.tolist(),
    }


# -----------------------------------------------------------------------------
# Run all (cohort, model) combinations
# -----------------------------------------------------------------------------
results = {}
for cohort_mode, cohort_label in [
    ("primary",   "Primary (full)"),
    ("no_preasd", "No pre-existing ASD"),
    ("ge12mo",    ">=12 mo follow-up"),
    ("both",      "Both filters"),
]:
    df_sub = filter_cohort(cohort_mode)
    X_enc, times, events = encode(df_sub)
    print(f"\n{'='*72}\nCohort: {cohort_label}  (n={len(df_sub)}, events={int(events.sum())})\n{'='*72}")
    cohort_results = {"n": len(df_sub), "n_events": int(events.sum()), "models": {}}
    oofs = {}
    for model_name in ("RSF", "GBSA", "DeepSurv"):
        if model_name == "DeepSurv" and not DEEP_OK:
            continue
        t0 = time.time()
        print(f"  [{model_name}] training...", flush=True)
        r = evaluate_model(model_name, X_enc, times, events)
        elapsed = time.time() - t0
        print(f"  [{model_name}] C={r['cindex_mean']:.3f}+/-{r['cindex_std']:.3f}  "
              f"AUC@12={r['auc_per_horizon'][12]:.3f}  AUC@60={r['auc_per_horizon'][60]:.3f}  "
              f"({elapsed:.0f}s)", flush=True)
        oofs[model_name] = np.array(r["oof_risk"])
        cohort_results["models"][model_name] = r
    # Spearman correlations between models in this cohort
    if len(oofs) >= 2:
        names = list(oofs.keys())
        corr = {}
        for i, a in enumerate(names):
            for b in names[i+1:]:
                va, vb = oofs[a], oofs[b]
                mask = ~(np.isnan(va) | np.isnan(vb))
                if mask.sum() > 5:
                    rho = pd.Series(va[mask]).rank().corr(pd.Series(vb[mask]).rank())
                    corr[f"{a}_vs_{b}"] = float(rho)
        cohort_results["spearman"] = corr
    results[cohort_mode] = {"label": cohort_label, **cohort_results}

# -----------------------------------------------------------------------------
# Side-by-side summary
# -----------------------------------------------------------------------------
print("\n\n" + "=" * 110)
print(f"{'Cohort':<24} {'Model':<10} {'n':>5} {'events':>7} {'C-index':>16} "
      f"{'AUC@12':>8} {'AUC@24':>8} {'AUC@36':>8} {'AUC@48':>8} {'AUC@60':>8}")
print("=" * 110)
for mode, r in results.items():
    for mname, mr in r["models"].items():
        print(f"{r['label']:<24} {mname:<10} {r['n']:>5d} {r['n_events']:>7d} "
              f"{mr['cindex_mean']:>8.3f} +/- {mr['cindex_std']:>4.3f}  "
              f"{mr['auc_per_horizon'][12]:>6.3f}  "
              f"{mr['auc_per_horizon'][24]:>6.3f}  "
              f"{mr['auc_per_horizon'][36]:>6.3f}  "
              f"{mr['auc_per_horizon'][48]:>6.3f}  "
              f"{mr['auc_per_horizon'][60]:>6.3f}")
    if "spearman" in r and r["spearman"]:
        for pair, rho in r["spearman"].items():
            print(f"{'  ↳ Spearman ' + pair:<24} {'':<10} {'':>5} {'':>7} {rho:>14.3f}")
print("=" * 110)

# Drop oof_risk from saved JSON to keep file small
for mode in results:
    for mname in results[mode]["models"]:
        results[mode]["models"][mname].pop("oof_risk", None)
with open(EXP / "sensitivity_multi_model.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {EXP / 'sensitivity_multi_model.json'}")
