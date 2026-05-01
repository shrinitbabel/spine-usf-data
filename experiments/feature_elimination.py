"""
Backward feature elimination for the RSF — start with all features
(67), iteratively drop the lowest-importance features by RSF split-count,
re-evaluate honest 10-fold CV C-index at each subset size. Find the
"knee" where parsimony stops costing discrimination.

Curse-of-dimensionality motivation: 122 events ÷ ~10 events-per-variable
rule of thumb = ~12 features. We currently have 67. This experiment finds
the empirical sweet spot.
"""

from __future__ import annotations
from pathlib import Path
import warnings, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Same data + preprocessing as fix_levels_and_retrain (with 6 level binaries)
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
times = df["time_surv"].values
events = df["REVERIFIED ASD"].astype(bool).values

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
numeric_like = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
                "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
                "post SVA","length of hospital stay (d)"]

X = df[present].copy()
for c in numeric_like:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
str_cats = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)


# --------------------------------------------------------------------------
# Step 1: rank all features by RSF split-count importance on full cohort
# --------------------------------------------------------------------------
def make_full_features(X_enc):
    """Apply rare-drop + PCA, return X_vals + feature names."""
    bin_cols = [c for c in X_enc.columns if set(np.unique(X_enc[c])) <= {0, 1}]
    rare = [c for c in bin_cols if X_enc[c].sum() < 5]
    keep = [c for c in X_enc.columns if c not in rare]
    Xk = X_enc[keep].copy()
    num_in = [c for c in numeric_like if c in keep]
    sc = StandardScaler().fit(Xk[num_in].values)
    n_pca = min(3, len(num_in))
    pca = PCA(n_components=n_pca, random_state=42).fit(sc.transform(Xk[num_in].values))
    pcs = pca.transform(sc.transform(Xk[num_in].values))
    df_pcs = pd.DataFrame(pcs, columns=[f"pca_num_{i+1}" for i in range(n_pca)],
                          index=Xk.index)
    final = pd.concat([Xk.drop(columns=num_in), df_pcs], axis=1)
    return final, num_in, sc, pca


X_final_full, num_in_full, sc_full, pca_full = make_full_features(X_encoded)
feat_full = X_final_full.columns.tolist()
print(f"Full feature space: {X_final_full.shape}")

y = Surv.from_arrays(event=events, time=times)
rsf_full = RandomSurvivalForest(
    n_estimators=200, min_samples_split=4, min_samples_leaf=12,
    max_features="sqrt", n_jobs=-1, random_state=42
).fit(X_final_full.values, y)

# Split-count importance
counts = np.zeros(len(feat_full))
for tree in rsf_full.estimators_:
    used = tree.tree_.feature
    used = used[used >= 0]
    np.add.at(counts, used, 1)
counts = counts / counts.sum()
ranked = pd.Series(counts, index=feat_full).sort_values(ascending=False)
print("\nTop 15 features by split-count importance:")
for i, (f, v) in enumerate(ranked.head(15).items(), 1):
    print(f"  {i:>2}. {f:55s}  {v:.4f}")


# --------------------------------------------------------------------------
# Step 2: honest 10-fold CV at multiple subset sizes
# --------------------------------------------------------------------------
def cv_eval_subset(top_k_feats):
    """
    Run honest CV using only the named top-k features.
    For PCA features, we keep them as-is (they're already engineered).
    For raw features that went into PCA (e.g. BMI), excluding them means
    they don't appear in numeric_like for the PCA step either.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis = []
    pca_cols = [f for f in top_k_feats if f.startswith("pca_num_")]
    raw_keep = [f for f in top_k_feats if not f.startswith("pca_num_")]
    n_pca = len(pca_cols)

    for tr, te in skf.split(X_encoded.values, events.astype(int)):
        if events[te].sum() == 0: continue
        Xtr_enc = X_encoded.iloc[tr]; Xte_enc = X_encoded.iloc[te]
        # Drop rare in train fold
        bin_cols = [c for c in Xtr_enc.columns if set(np.unique(Xtr_enc[c])) <= {0,1}]
        rare = [c for c in bin_cols if Xtr_enc[c].sum() < 5]
        keep = [c for c in Xtr_enc.columns if c not in rare]
        Xtr_k = Xtr_enc[keep]; Xte_k = Xte_enc[keep]
        num_in = [c for c in numeric_like if c in keep]
        sc = StandardScaler().fit(Xtr_k[num_in].values)
        n_p = min(3, len(num_in))
        pca = PCA(n_components=n_p, random_state=42).fit(sc.transform(Xtr_k[num_in].values))
        def transform(Xv):
            pcs = pca.transform(sc.transform(Xv[num_in].values))
            rest = Xv.drop(columns=num_in).astype(float).values
            full = np.concatenate([rest, pcs], axis=1)
            cols = [c for c in Xv.columns if c not in num_in] + [f"pca_num_{i+1}" for i in range(n_p)]
            return pd.DataFrame(full, columns=cols, index=Xv.index)
        Xtr_full = transform(Xtr_k); Xte_full = transform(Xte_k)
        # Subset to requested features
        sub = [f for f in top_k_feats if f in Xtr_full.columns]
        if not sub: continue
        Xtr_sub = Xtr_full[sub].values
        Xte_sub = Xte_full[sub].values
        rsf = RandomSurvivalForest(
            n_estimators=200, min_samples_split=4, min_samples_leaf=12,
            max_features="sqrt", n_jobs=-1, random_state=42
        ).fit(Xtr_sub, Surv.from_arrays(event=events[tr], time=times[tr]))
        ci = concordance_index_censored(events[te], times[te], rsf.predict(Xte_sub))[0]
        cis.append(ci)
    cis = np.array(cis)
    return cis


# Test multiple subset sizes
SUBSET_SIZES = [67, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 4]
print("\n=== Honest CV C-index vs feature subset size ===")
print(f"  {'k':>4s}  {'C mean ± std':>16s}  {'min':>6s}  {'max':>6s}  {'(features)':<10s}")
print("  " + "-" * 70)

results = []
for k in SUBSET_SIZES:
    sub = ranked.head(k).index.tolist()
    cis = cv_eval_subset(sub)
    if len(cis):
        mean_c = cis.mean(); std_c = cis.std()
        results.append({"k": k, "mean": mean_c, "std": std_c,
                        "min": cis.min(), "max": cis.max(),
                        "features": sub})
        print(f"  {k:>4d}  {mean_c:.3f} ± {std_c:.3f}  "
              f"{cis.min():.3f}  {cis.max():.3f}")

# Find best
best = max(results, key=lambda r: r["mean"])
print(f"\nBest by mean C-index: k={best['k']}, C={best['mean']:.3f} ± {best['std']:.3f}")
print(f"Features in best subset:")
for f in best["features"]:
    print(f"  - {f}")

# Save
import json
with open(ROOT / "experiments" / "feature_elimination_results.json", "w") as f:
    json.dump({
        "results": [{"k": r["k"], "mean": float(r["mean"]),
                     "std": float(r["std"]), "min": float(r["min"]),
                     "max": float(r["max"]), "features": r["features"]}
                    for r in results],
        "best_k": best["k"],
        "split_count_ranking": [{"feature": f, "importance": float(v)}
                                 for f, v in ranked.items()],
    }, f, indent=2)
print(f"\nSaved -> experiments/feature_elimination_results.json")
