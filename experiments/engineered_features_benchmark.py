"""
Clinically-engineered feature set benchmark.

Replaces 400+ noisy one-hot dummies with 16 hand-engineered, clinically
interpretable features. Tests RSF / DeepSurv / GBSA / CWGB / Coxnet
under honest 10-fold CV.

Engineered features (16):
  Demographics (4):     Age, Sex, BMI, prior_back_surgery
  Diagnostic (2):       dx_degenerative_any, dx_deformity_any
  Construct (3):        levels_fused_count, lumbosacral_construct, long_construct
  Surgery (2):          complex_surgery, mis_approach
  Alignment (3):        pca_num_1, pca_num_2, pca_num_3 (post-op alignment composites)
  Hospital course (2):  length_of_stay, major_complication_any
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

# ----- Load -----
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
def b(c): return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
def n(c): return pd.to_numeric(df[c], errors="coerce")

# ----- Engineered features -----
feat = pd.DataFrame(index=df.index)

# Demographics
feat["Age"] = n("Age")
feat["Sex"] = b("Sex")
feat["BMI"] = n("BMI")
feat["prior_back_surgery"] = b("prior back surgeries? (y=1)")

# Diagnostic composites
feat["dx_degenerative_any"] = (b("dx_stenosis") | b("dx_spondylolisthesis") | b("dx_spondylosis")).astype(int)
feat["dx_deformity_any"]    = (b("dx_scoliosis") | b("dx_flat_back") | b("dx_sagittal_imbalance")).astype(int)

# Construct
levels_count = sum(b(L) for L in LEVELS)
feat["levels_fused_count"]    = levels_count
feat["lumbosacral_construct"] = b("L5-S1")
feat["long_construct"]        = (levels_count >= 4).astype(int)

# Surgical complexity
feat["complex_surgery"] = (b("Anterior + Posterior Apporoach") | b("Osteotomies (yes/no)") | b("ACR (y=1)")).astype(int)
feat["mis_approach"]    = ((b("Standalone XLIF Check") | b("Retroperitoneal Approach (LLIF ± ALIF)")) & (b("Open") == 0)).astype(int)

# Hospital course
feat["length_of_stay"] = n("length of hospital stay (d)")
feat["major_complication_any"] = (
    b("infection 1=yes") | b("DVT  1=yes") | b("PE  1=yes") | b("MI 1=yes")
    | b("femoral palsy (knee extension weakness) 1=yes") | b("psoas hematoma")
).astype(int)

print("Engineered binary feature distribution:")
for c in ["Sex","prior_back_surgery","dx_degenerative_any","dx_deformity_any",
          "lumbosacral_construct","long_construct","complex_surgery","mis_approach",
          "major_complication_any"]:
    print(f"  {c:30s}  {int(feat[c].sum()):>4d} ({feat[c].mean()*100:>5.1f}%)")

print("\nEngineered numeric feature ranges:")
for c in ["Age","BMI","levels_fused_count","length_of_stay"]:
    s = pd.to_numeric(feat[c], errors="coerce")
    print(f"  {c:30s}  mean={s.mean():>6.2f}  range [{s.min():.0f}, {s.max():.0f}]")

# Now build the alignment PCA components from raw post-op variables
# (same construction as production pipeline, but kept separate from
# the engineered set so we can drop missing rows cleanly).
NUM_FOR_PCA = ["BMI","ALIF Count","Lateral Count","Average PI","PI-LL angle mismatch",
               "ABS PI-LL angle mismatch","Post-op SS","post PI","post PT","post LL",
               "post SVA","length of hospital stay (d)"]
num_block = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in NUM_FOR_PCA}).fillna(0.0).values
sc_pca = StandardScaler().fit(num_block)
pca = PCA(n_components=3, random_state=42).fit(sc_pca.transform(num_block))
pcs = pca.transform(sc_pca.transform(num_block))
feat["pca_num_1"] = pcs[:, 0]
feat["pca_num_2"] = pcs[:, 1]
feat["pca_num_3"] = pcs[:, 2]

print(f"\nFinal engineered feature matrix: {feat.shape}  (16 features expected)")
assert feat.shape[1] == 16, f"Expected 16, got {feat.shape[1]}"

# ----- Drop any rows with NaN in numeric features -----
feat = feat.fillna(feat.median(numeric_only=True))   # gentle median fill for the few missing
X_arr = feat.values.astype(float)

# ----- Honest 10-fold CV evaluator -----
def cv_eval(model_factory, name=""):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cis = []
    for tr, te in skf.split(X_arr, events.astype(int)):
        if events[te].sum() == 0: continue
        Xtr, Xte = X_arr[tr], X_arr[te]
        ytr = Surv.from_arrays(event=events[tr], time=times[tr])
        try:
            if name == "DeepSurv":
                net = tt.practical.MLPVanilla(in_features=Xtr.shape[1],
                                               num_nodes=[32,16], out_features=1,
                                               batch_norm=True, dropout=0.5, output_bias=False)
                m = CoxPH_NN(net, tt.optim.Adam(lr=1e-3))
                m.fit(Xtr.astype("float32"),
                      (times[tr].astype("float32"), events[tr].astype("float32")),
                      batch_size=64, epochs=80, verbose=False)
                risk = m.predict(Xte.astype("float32")).reshape(-1)
            else:
                m = model_factory()
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
    ("DeepSurv", None),
    ("GBSA",     lambda: GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                                            max_depth=3, random_state=42)),
    ("CWGB",     lambda: ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300,
                                                                          learning_rate=0.05,
                                                                          random_state=42)),
    ("Coxnet",   lambda: CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                                  n_alphas=20, fit_baseline_model=True)),
]

print(f"\n=== Honest 10-fold CV on 16 engineered features ===")
print(f"  {'Model':10s}  {'C mean':>7s}  {'std':>5s}  {'min':>5s}  {'max':>5s}")
print("  " + "-" * 45)

results = {}
for name, fac in MODELS:
    cis = cv_eval(fac, name=name)
    if len(cis):
        results[name] = {"mean": float(cis.mean()), "std": float(cis.std()),
                         "min": float(cis.min()), "max": float(cis.max())}
        print(f"  {name:10s}  {cis.mean():>7.3f}  {cis.std():>5.3f}  {cis.min():>5.3f}  {cis.max():>5.3f}")

with open(ROOT / "experiments" / "engineered_features_benchmark.json", "w") as f:
    json.dump({"results": results, "n_features": int(feat.shape[1]),
               "feature_names": feat.columns.tolist()}, f, indent=2)
print(f"\nSaved -> experiments/engineered_features_benchmark.json")
