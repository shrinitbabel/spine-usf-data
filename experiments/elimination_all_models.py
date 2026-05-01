"""
Feature elimination across ALL candidate models, on top of the
variant-C feature set (single levels_fused_count integer instead of
six per-level binaries — that variant alone got RSF to C ≈ 0.601).

For each subset size k ∈ {full, 50, 30, 20, 15, 12, 10, 8}, fit each of
RSF / DeepSurv / GBSA / Coxnet / CWGB under honest 10-fold CV and report
mean C ± std.

Ranking is once-and-for-all by RSF split-count importance on the full
feature space (cheap and consistent across model comparisons).
"""

from __future__ import annotations
from pathlib import Path
import warnings, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.ensemble import (RandomSurvivalForest, GradientBoostingSurvivalAnalysis,
                              ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import torch, torchtuples as tt
from pycox.models import CoxPH as CoxPH_NN

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Build variant-C feature space (levels_fused_count integer)
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

LEVELS = ["T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"]

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
present = [c for c in base_cols if c in df.columns]
numeric_like = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
                "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
                "post SVA","length of hospital stay (d)","levels_fused_count"]

X = df[present].copy()
for c in numeric_like:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
# Add levels_fused_count
X["levels_fused_count"] = sum(df[L].astype(int) for L in LEVELS).astype(int)

str_cats = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)
print(f"Variant-C encoded matrix: {X_encoded.shape}")

# --------------------------------------------------------------------------
# Rank features once via full-cohort RSF split-count
# --------------------------------------------------------------------------
def make_full(X_enc):
    bin_cols = [c for c in X_enc.columns if set(np.unique(X_enc[c])) <= {0,1}]
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
    return final, num_in

X_full, num_in_full = make_full(X_encoded)
feat_full = X_full.columns.tolist()
print(f"Full feature space (post-PCA): {X_full.shape}")

y = Surv.from_arrays(event=events, time=times)
rsf_rank = RandomSurvivalForest(n_estimators=200, min_samples_split=4,
                                 min_samples_leaf=12, max_features="sqrt",
                                 n_jobs=-1, random_state=42).fit(X_full.values, y)
counts = np.zeros(len(feat_full))
for tree in rsf_rank.estimators_:
    used = tree.tree_.feature
    used = used[used >= 0]
    np.add.at(counts, used, 1)
ranked = pd.Series(counts/counts.sum(), index=feat_full).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Per-fold CV evaluator that builds the pipeline and subsets to top-k feats
# --------------------------------------------------------------------------
def cv_eval(model_factory, top_k_feats, name=""):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis = []
    for tr, te in skf.split(X_encoded.values, events.astype(int)):
        if events[te].sum() == 0: continue
        Xtr_enc = X_encoded.iloc[tr]; Xte_enc = X_encoded.iloc[te]
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
        sub = [f for f in top_k_feats if f in Xtr_full.columns]
        if not sub: continue
        Xtr = Xtr_full[sub].values.astype(float)
        Xte = Xte_full[sub].values.astype(float)
        ytr = Surv.from_arrays(event=events[tr], time=times[tr])

        try:
            m = model_factory()
            if name == "DeepSurv":
                net = tt.practical.MLPVanilla(in_features=Xtr.shape[1],
                                               num_nodes=[32,16], out_features=1,
                                               batch_norm=True, dropout=0.5,
                                               output_bias=False)
                m = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
                m.fit(Xtr.astype("float32"),
                      (times[tr].astype("float32"), events[tr].astype("float32")),
                      batch_size=64, epochs=80, verbose=False)
                risk = m.predict(Xte.astype("float32")).reshape(-1)
            else:
                m.fit(Xtr, ytr)
                risk = m.predict(Xte)
            ci = concordance_index_censored(events[te], times[te], risk)[0]
            cis.append(ci)
        except Exception as ex:
            pass
    return np.array(cis)


MODELS = [
    ("RSF",      lambda: RandomSurvivalForest(n_estimators=200, min_samples_split=4,
                                                min_samples_leaf=12, max_features="sqrt",
                                                n_jobs=-1, random_state=42)),
    ("DeepSurv", None),  # special case in cv_eval
    ("GBSA",     lambda: GradientBoostingSurvivalAnalysis(n_estimators=300,
                                                            learning_rate=0.05,
                                                            max_depth=3, random_state=42)),
    ("CWGB",     lambda: ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300,
                                                                          learning_rate=0.05,
                                                                          random_state=42)),
    ("Coxnet",   lambda: CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                                  n_alphas=20, fit_baseline_model=True)),
]
SUBSET_SIZES = [len(feat_full), 50, 30, 20, 15, 12, 10]

print(f"\n=== Honest 10-fold CV: variant-C base, all models ×  subset sizes ===")
print(f"  {'k':>4s}  {'RSF':>13s}  {'DeepSurv':>13s}  {'GBSA':>13s}  {'CWGB':>13s}  {'Coxnet':>13s}")
print("  " + "-" * 80)

results = {}
for k in SUBSET_SIZES:
    sub = ranked.head(k).index.tolist()
    row = {"k": k}
    line = f"  {k:>4d}  "
    for name, fac in MODELS:
        cis = cv_eval(fac, sub, name=name)
        if len(cis):
            mean_c = cis.mean(); std_c = cis.std()
            row[name] = {"mean": float(mean_c), "std": float(std_c),
                         "min": float(cis.min()), "max": float(cis.max())}
            line += f"{mean_c:.3f} ± {std_c:.3f}  "
        else:
            row[name] = None
            line += f"{'fail':>13s}  "
    print(line)
    results[k] = row

with open(ROOT / "experiments" / "elimination_all_models_results.json", "w") as f:
    json.dump({"results": list(results.values()),
               "split_count_ranking": [{"feature": f, "importance": float(v)}
                                        for f, v in ranked.items()]},
              f, indent=2)
print(f"\nSaved -> experiments/elimination_all_models_results.json")
