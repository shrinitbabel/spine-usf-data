"""
Quick A/B/C: how does level-of-fusion encoding affect honest CV C-index?

  A — no level features at all (the 'old 0.598' regime, where level
      binaries were dead-zero and effectively ignored)
  B — 6 individual level binaries (current production, C ≈ 0.583)
  C — single levels_fused_count integer feature
"""

from __future__ import annotations
from pathlib import Path
import warnings
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

base = [
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
numeric_like = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
                "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
                "post SVA","length of hospital stay (d)"]


def build(variant: str):
    cols = [c for c in base if c in df.columns]
    if variant == "B":
        cols = cols + LEVELS
    X = df[cols].copy()
    for c in numeric_like:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if variant == "C":
        X["levels_fused_count"] = sum(df[L].astype(int) for L in LEVELS).astype(int)
    str_cats = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=str_cats, drop_first=True).fillna(0.0)
    return X


def cv_eval(name, X_enc):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis = []
    for fold, (tr, te) in enumerate(skf.split(X_enc.values, events.astype(int)), 1):
        if events[te].sum() == 0: continue
        Xtr_raw = X_enc.iloc[tr]; Xte_raw = X_enc.iloc[te]
        bin_cols = [c for c in Xtr_raw.columns if set(np.unique(Xtr_raw[c])) <= {0,1}]
        rare = [c for c in bin_cols if Xtr_raw[c].sum() < 5]
        keep = [c for c in Xtr_raw.columns if c not in rare]
        Xtr = Xtr_raw[keep]; Xte = Xte_raw[keep]
        num_in = [c for c in numeric_like if c in keep]
        sc = StandardScaler().fit(Xtr[num_in].values)
        n_pca = min(3, len(num_in))
        pca = PCA(n_components=n_pca, random_state=42).fit(sc.transform(Xtr[num_in].values))
        def transform(Xv):
            pcs = pca.transform(sc.transform(Xv[num_in].values))
            rest = Xv.drop(columns=num_in).astype(float).values
            return np.concatenate([rest, pcs], axis=1)
        Xtr_f = transform(Xtr); Xte_f = transform(Xte)
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_split=4,
                                    min_samples_leaf=12, max_features="sqrt",
                                    n_jobs=-1, random_state=42).fit(
            Xtr_f, Surv.from_arrays(event=events[tr], time=times[tr]))
        ci = concordance_index_censored(events[te], times[te], rsf.predict(Xte_f))[0]
        cis.append(ci)
    cis = np.array(cis)
    print(f"  {name:55s}  n_feats={X_enc.shape[1]}  C={cis.mean():.3f} ± {cis.std():.3f}  range [{cis.min():.3f}, {cis.max():.3f}]")
    return cis


print("=== Level-encoding A/B/C ===\n")
for variant, label in [
    ("A", "A. NO level features (same as 'old 0.598' regime)"),
    ("B", "B. 6 individual level binaries (current)"),
    ("C", "C. single levels_fused_count integer"),
]:
    X = build(variant)
    cv_eval(label, X)
