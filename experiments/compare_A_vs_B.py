"""
Head-to-head comparison of Option A vs Option B for the manuscript.

  A = Variant-C feature space (single levels_fused_count integer, ~62
      features after rare-drop + PCA) + RSF
  B = 16 clinically engineered features + Componentwise Gradient Boost

Reports for each:
  - C-index (per-fold + pooled OOF)
  - IBS over fold-specific time grid
  - Time-dependent AUC at 12 / 24 / 36 / 48 / 60 months
  - Pooled ECE at the same horizons
  - SHAP top-15 features (saved as separate beeswarm PNGs)
"""

from __future__ import annotations
from pathlib import Path
import warnings, json, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.ensemble import (RandomSurvivalForest,
                              ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.metrics import (concordance_index_censored,
                             integrated_brier_score, cumulative_dynamic_auc)
from lifelines import KaplanMeierFitter
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "figures"; FIG.mkdir(exist_ok=True)
EXP  = ROOT / "experiments"

mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "sans-serif", "font.size": 10})

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "cleaned_data.csv")
df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
df["REVERIFIED ASD"] = df["REVERIFIED ASD"].fillna(0).astype(int)
df["time_surv"] = np.where(df["REVERIFIED ASD"] == 1,
                            df["Time Until ASD Diagnosis (months)"],
                            df["Time Without_ASD (months)"])
df = df.dropna(subset=["time_surv"]).reset_index(drop=True)
df["time_surv"] = df["time_surv"].astype(float)
times  = df["time_surv"].values
events = df["REVERIFIED ASD"].astype(bool).values
N = len(df)

LEVELS = ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]
def b(c): return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
def n(c): return pd.to_numeric(df[c], errors="coerce")

# --------------------------------------------------------------------------
# Build A (Variant-C: replace 6 binaries with single levels_fused_count)
# --------------------------------------------------------------------------
print("Building Option A (Variant C + RSF)...")
base_cols = [
    "Sex","Age","BMI","prior back surgeries? (y=1)",
    "dx_adjacent_segment","dx_spondylolisthesis","dx_spondylosis","dx_stenosis",
    "dx_scoliosis","dx_flat_back","dx_sagittal_imbalance","dx_post_laminectomy","dx_deformity",
    "Case/Type of Surgery","Additional Procedures w/in surgery",
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
NUM_A = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
         "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
         "post SVA","length of hospital stay (d)","levels_fused_count"]
present = [c for c in base_cols if c in df.columns]
XA = df[present].copy()
for c in NUM_A:
    if c in XA.columns:
        XA[c] = pd.to_numeric(XA[c], errors="coerce")
XA["levels_fused_count"] = sum(b(L) for L in LEVELS).astype(int)
XA = pd.get_dummies(XA, columns=XA.select_dtypes(include=["object"]).columns.tolist(),
                     drop_first=True).fillna(0.0)


# --------------------------------------------------------------------------
# Build B (16 engineered features)
# --------------------------------------------------------------------------
print("Building Option B (16 engineered + CWGB)...")
XB = pd.DataFrame(index=df.index)
XB["Age"] = n("Age"); XB["Sex"] = b("Sex"); XB["BMI"] = n("BMI")
XB["prior_back_surgery"] = b("prior back surgeries? (y=1)")
XB["dx_degenerative_any"] = (b("dx_stenosis") | b("dx_spondylolisthesis") | b("dx_spondylosis")).astype(int)
XB["dx_deformity_any"]    = (b("dx_scoliosis") | b("dx_flat_back") | b("dx_sagittal_imbalance")).astype(int)
levels_count = sum(b(L) for L in LEVELS)
XB["levels_fused_count"]    = levels_count
XB["lumbosacral_construct"] = b("L5-S1")
XB["long_construct"]        = (levels_count >= 4).astype(int)
XB["complex_surgery"] = (b("Anterior + Posterior Apporoach") | b("Osteotomies (yes/no)") | b("ACR (y=1)")).astype(int)
XB["mis_approach"]    = ((b("Standalone XLIF Check") | b("Retroperitoneal Approach (LLIF ± ALIF)")) & (b("Open") == 0)).astype(int)
XB["length_of_stay"] = n("length of hospital stay (d)")
XB["major_complication_any"] = (
    b("infection 1=yes") | b("DVT  1=yes") | b("PE  1=yes") | b("MI 1=yes")
    | b("femoral palsy (knee extension weakness) 1=yes") | b("psoas hematoma")
).astype(int)
NUM_FOR_PCA = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
               "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
               "post SVA","length of hospital stay (d)"]
nb = pd.DataFrame({c: n(c) for c in NUM_FOR_PCA}).fillna(0.0).values
sc_pca_full = StandardScaler().fit(nb)
pca_full = PCA(n_components=3, random_state=42).fit(sc_pca_full.transform(nb))
pcs = pca_full.transform(sc_pca_full.transform(nb))
XB["pca_num_1"] = pcs[:, 0]; XB["pca_num_2"] = pcs[:, 1]; XB["pca_num_3"] = pcs[:, 2]
XB = XB.fillna(XB.median(numeric_only=True))


HORIZONS = [12, 24, 36, 48, 60]


# --------------------------------------------------------------------------
# Per-fold preprocessing for Option A
# --------------------------------------------------------------------------
def fit_pipeline_A(X_tr):
    bin_cols = [c for c in X_tr.columns if set(np.unique(X_tr[c])) <= {0,1}]
    rare = [c for c in bin_cols if X_tr[c].sum() < 5]
    keep = [c for c in X_tr.columns if c not in rare]
    Xk = X_tr[keep]
    num_in = [c for c in NUM_A if c in keep]
    sc = StandardScaler().fit(Xk[num_in].values)
    n_pca = min(3, len(num_in))
    pca = PCA(n_components=n_pca, random_state=42).fit(sc.transform(Xk[num_in].values))
    def transform(Xv):
        Xv = Xv[keep]
        pcs = pca.transform(sc.transform(Xv[num_in].values))
        rest = Xv.drop(columns=num_in).astype(float).values
        return np.concatenate([rest, pcs], axis=1), [c for c in Xv.columns if c not in num_in] + [f"pca_num_{i+1}" for i in range(n_pca)]
    return transform


def surv_at(model, X, horizons):
    funcs = model.predict_survival_function(X, return_array=False)
    out = np.zeros((len(funcs), len(horizons)))
    for i, fn in enumerate(funcs):
        out[i] = np.interp(horizons, fn.x, fn.y, left=1.0, right=fn.y[-1])
    return out


# --------------------------------------------------------------------------
# Honest 10-fold CV with all metrics for one model+feature combo
# --------------------------------------------------------------------------
def run_cv(X_arr_or_df, model_factory, name, is_option_A=False):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis, ibss = [], []
    auc_per_fold = {h: [] for h in HORIZONS}
    OOF_S = np.full((N, len(HORIZONS)), np.nan)
    OOF_risk = np.full(N, np.nan)
    for tr, te in skf.split(np.zeros(N), events.astype(int)):
        if events[te].sum() == 0: continue
        if is_option_A:
            transform = fit_pipeline_A(X_arr_or_df.iloc[tr])
            Xtr, _ = transform(X_arr_or_df.iloc[tr])
            Xte, _ = transform(X_arr_or_df.iloc[te])
        else:
            Xtr, Xte = X_arr_or_df.values[tr], X_arr_or_df.values[te]
        ytr = Surv.from_arrays(event=events[tr], time=times[tr])
        yte = Surv.from_arrays(event=events[te], time=times[te])
        m = model_factory().fit(Xtr.astype(float), ytr)
        risk = m.predict(Xte.astype(float))
        OOF_risk[te] = risk
        OOF_S[te] = surv_at(m, Xte.astype(float), HORIZONS)
        ci = concordance_index_censored(events[te], times[te], risk)[0]
        cis.append(ci)
        # IBS
        t_max = min(times[tr].max(), times[te].max()) - 1e-3
        t_min = max(times[tr].min(), times[te].min()) + 1e-3
        ibs_grid = np.linspace(t_min, t_max, 10)
        try:
            S_ibs = surv_at(m, Xte.astype(float), list(ibs_grid))
            ibss.append(integrated_brier_score(ytr, yte, S_ibs, times=ibs_grid))
        except Exception:
            ibss.append(float("nan"))
        # Time-AUC
        valid_h = [h for h in HORIZONS if t_min < h < t_max]
        if valid_h:
            try:
                aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, np.asarray(valid_h, dtype=float))
                aucs = np.atleast_1d(aucs)
                for h, a in zip(valid_h, aucs):
                    auc_per_fold[h].append(float(a))
            except Exception:
                pass

    cis = np.array(cis); ibss = np.array(ibss)
    print(f"\n  {name}")
    print(f"    C-index  = {cis.mean():.3f} ± {cis.std():.3f}  range [{cis.min():.3f}, {cis.max():.3f}]")
    print(f"    IBS      = {ibss.mean():.3f} ± {ibss.std():.3f}")
    print(f"    Time-dependent AUC by horizon:")
    for h in HORIZONS:
        v = np.array(auc_per_fold[h])
        if len(v):
            print(f"      {h:>3} mo  AUC = {v.mean():.3f} ± {v.std():.3f}")

    # Pooled ECE per horizon (KM-binned)
    print(f"    Pooled ECE by horizon:")
    ece_dict = {}
    for hi, h in enumerate(HORIZONS):
        S = OOF_S[:, hi]
        valid = ~np.isnan(S)
        preds = S[valid]; t = times[valid]; e = events[valid]
        try:
            bin_id = pd.qcut(pd.Series(preds).rank(method="first"), q=10,
                              labels=False, duplicates="drop")
        except ValueError:
            ece_dict[h] = float("nan"); continue
        rows = []
        for bb in sorted(np.unique(bin_id)):
            mask = bin_id == bb
            if mask.sum() < 3: continue
            try:
                km = KaplanMeierFitter().fit(t[mask], event_observed=e[mask])
                obs = float(km.predict(h))
            except Exception:
                obs = float("nan")
            rows.append({"n": int(mask.sum()),
                         "pred_mean": float(np.mean(preds[mask])), "obs": obs})
        bdf = pd.DataFrame(rows).dropna(subset=["obs"])
        if len(bdf) == 0:
            ece_dict[h] = float("nan"); continue
        c = bdf["n"].values.astype(float)
        ece = float(np.sum((c / c.sum()) * np.abs(bdf["pred_mean"].values - bdf["obs"].values)))
        ece_dict[h] = ece
        print(f"      {h:>3} mo  ECE = {ece:.4f}")

    valid = ~np.isnan(OOF_risk)
    pooled_ci = concordance_index_censored(events[valid], times[valid], OOF_risk[valid])[0]
    print(f"    Pooled OOF C = {pooled_ci:.3f}")
    return {
        "cindex_mean": float(cis.mean()), "cindex_std": float(cis.std()),
        "cindex_min": float(cis.min()), "cindex_max": float(cis.max()),
        "ibs_mean": float(ibss.mean()), "ibs_std": float(ibss.std()),
        "auc_per_horizon": {h: {"mean": float(np.mean(auc_per_fold[h])),
                                  "std": float(np.std(auc_per_fold[h]))} for h in HORIZONS if auc_per_fold[h]},
        "ece_per_horizon": ece_dict,
        "pooled_oof_cindex": float(pooled_ci),
    }


print("\n========== HONEST CV ==========")
res_A = run_cv(XA, lambda: RandomSurvivalForest(n_estimators=200, min_samples_split=4,
                                                  min_samples_leaf=12, max_features="sqrt",
                                                  n_jobs=-1, random_state=42),
               "Option A — Variant C + RSF (~62 features)", is_option_A=True)
res_B = run_cv(XB, lambda: ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300,
                                                                            learning_rate=0.05,
                                                                            random_state=42),
               "Option B — 16 engineered + CWGB", is_option_A=False)


# --------------------------------------------------------------------------
# SHAP for each (full-cohort fit)
# --------------------------------------------------------------------------
print("\n========== SHAP COMPARISON ==========")
print("Fitting Option A (RSF) on full cohort + SHAP...")
transform_A_final = fit_pipeline_A(XA)
X_A_arr, A_feats = transform_A_final(XA)
y_full = Surv.from_arrays(event=events, time=times)
rsf_A = RandomSurvivalForest(n_estimators=200, min_samples_split=4,
                              min_samples_leaf=12, max_features="sqrt",
                              n_jobs=-1, random_state=42).fit(X_A_arr, y_full)
np.random.seed(42)
bg_A = X_A_arr[np.random.choice(len(X_A_arr), 50, replace=False)]
expl_A = shap.PermutationExplainer(rsf_A.predict, bg_A)
print("  Computing SHAP for Option A (this takes ~5 min)...")
shap_A = expl_A(X_A_arr.astype(float))
print(f"  Option A SHAP shape: {shap_A.values.shape}")

print("\nFitting Option B (CWGB) on full cohort + SHAP...")
X_B_arr = XB.values.astype(float)
B_feats = list(XB.columns)
cwgb_B = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                                          random_state=42).fit(X_B_arr, y_full)
bg_B = X_B_arr[np.random.choice(len(X_B_arr), 50, replace=False)]
expl_B = shap.PermutationExplainer(cwgb_B.predict, bg_B)
print("  Computing SHAP for Option B (this takes ~3 min)...")
shap_B = expl_B(X_B_arr)
print(f"  Option B SHAP shape: {shap_B.values.shape}")

# Save SHAP arrays
np.savez_compressed(EXP / "shap_compare_A_B.npz",
                    shap_A=shap_A.values, X_A=X_A_arr, feats_A=np.array(A_feats, dtype=object),
                    shap_B=shap_B.values, X_B=X_B_arr, feats_B=np.array(B_feats, dtype=object))


# --------------------------------------------------------------------------
# SHAP top-15 for each — printable + side-by-side figure
# --------------------------------------------------------------------------
def top15(sv, feats):
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[-15:][::-1]
    return [(feats[i], float(mean_abs[i])) for i in order]

print("\nOption A — top 15 features by mean |SHAP|:")
for i, (f, v) in enumerate(top15(shap_A.values, A_feats), 1):
    print(f"  {i:>2}. {f[:55]:55s}  {v:.4f}")

print("\nOption B — top 15 features by mean |SHAP|:")
for i, (f, v) in enumerate(top15(shap_B.values, B_feats), 1):
    print(f"  {i:>2}. {f[:55]:55s}  {v:.4f}")


# --------------------------------------------------------------------------
# Side-by-side SHAP summary figures
# --------------------------------------------------------------------------
def beeswarm_panel(ax, sv, X_arr, feats, title):
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:]
    rng = np.random.RandomState(42)
    cmap = plt.get_cmap("coolwarm")
    for row, fi in enumerate(top_idx):
        s = sv[:, fi]; v = X_arr[:, fi].astype(float)
        if v.std() > 0:
            v_norm = (v - np.nanpercentile(v, 5)) / max(np.nanpercentile(v, 95) - np.nanpercentile(v, 5), 1e-9)
        else:
            v_norm = np.full_like(v, 0.5)
        v_norm = np.clip(v_norm, 0, 1)
        # density-aware jitter
        counts, edges = np.histogram(s, bins=30)
        bin_idx = np.clip(np.digitize(s, edges) - 1, 0, len(counts) - 1)
        density = counts[bin_idx] / max(counts.max(), 1)
        jitter = rng.normal(0, 0.16, size=len(s)) * density
        ax.scatter(s, np.full(len(s), row) + jitter, c=cmap(v_norm), s=14, alpha=0.75,
                   edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_yticks(range(15))
    ax.set_yticklabels([feats[i][:42] for i in top_idx])
    ax.set_xlabel("SHAP value (contribution to risk score)")
    ax.set_title(title, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
beeswarm_panel(axes[0], shap_A.values, X_A_arr, A_feats,
               f"Option A — Variant C + RSF (~62 features)\nC = {res_A['cindex_mean']:.3f} ± {res_A['cindex_std']:.3f}")
beeswarm_panel(axes[1], shap_B.values, X_B_arr, B_feats,
               f"Option B — 16 engineered + CWGB\nC = {res_B['cindex_mean']:.3f} ± {res_B['cindex_std']:.3f}")
fig.suptitle("SHAP top-15 feature contributions, Option A vs Option B", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(FIG / "fig31_shap_compare_A_vs_B.png")
plt.close(fig)
print(f"\nSaved fig31_shap_compare_A_vs_B.png")

with open(EXP / "compare_A_vs_B_results.json", "w") as f:
    json.dump({"option_A": res_A, "option_B": res_B,
               "n_features_A": X_A_arr.shape[1],
               "n_features_B": X_B_arr.shape[1]}, f, indent=2)
print("Saved -> experiments/compare_A_vs_B_results.json")
