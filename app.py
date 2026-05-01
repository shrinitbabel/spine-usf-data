import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

import torch
import torchtuples as tt
from pycox.models import CoxPH as CoxPH_NN

import shap


# Maps each Pydantic field to the column header the trained model actually uses.
# Front-end keeps sending JSON-friendly underscored names; we translate to the
# original headers (spaces, dashes, parens, double spaces — all preserved
# exactly as they appear in cleaned_data.csv after .str.strip()).
FIELD_TO_COL = {
    "Sex": "Sex",
    "Age": "Age",
    "BMI": "BMI",
    "prior_back_surgeries": "prior back surgeries? (y=1)",

    "dx_adjacent_segment": "dx_adjacent_segment",
    "dx_spondylolisthesis": "dx_spondylolisthesis",
    "dx_spondylosis": "dx_spondylosis",
    "dx_stenosis": "dx_stenosis",
    "dx_scoliosis": "dx_scoliosis",
    "dx_flat_back": "dx_flat_back",
    "dx_sagittal_imbalance": "dx_sagittal_imbalance",
    "dx_post_laminectomy": "dx_post_laminectomy",
    "dx_deformity": "dx_deformity",

    "T12_L1": "T12-L1",
    "L1_L2": "L1-L2",
    "L2_L3": "L2-L3",
    "L3_L4": "L3-L4",
    "L4_L5": "L4-L5",
    "L5_S1": "L5-S1",

    "Open": "Open",
    "Perc_screws": "Perc screws?",
    "Lateral_Count": "Lateral Count",
    "ALIF_Count": "ALIF Count",
    "ACR": "ACR (y=1)",

    "Average_PI": "Average PI",
    "PI_LL_mismatch": "PI-LL angle mismatch",
    "ABS_PI_LL_mismatch": "ABS PI-LL angle mismatch",
    "PI_gt_50": "(1 = PI>50)",

    "post_LL": "post LL",
    "post_SVA": "post SVA",
    "post_PI": "post PI",
    "post_PT": "post PT",
    "Post_op_SS": "Post-op SS",

    "LOS": "length of hospital stay (d)",
    "Open_Check_V2": "Open Check V2",
    "Standalone_XLIF_Check": "Standalone XLIF Check",
    "Retroperitoneal_Approach_LLIF_ALIF": "Retroperitoneal Approach (LLIF ± ALIF)",
    "Anterior_Posterior_Apporoach": "Anterior + Posterior Apporoach",
    "Osteotomies_yes_no": "Osteotomies (yes/no)",

    "infection_1_yes": "infection 1=yes",
    "DVT_1_yes": "DVT  1=yes",
    "PE_1_yes": "PE  1=yes",
    "MI_1_yes": "MI 1=yes",
    "femoral_palsy_1_yes": "femoral palsy (knee extension weakness) 1=yes",
    "hip_flexion_weakness_1_yes": "hip flexion weakness (iliopsoas weakness)  1=yes",
    "acute_thigh_paresthesia": "acute thigh paresthesia (immediate post op)",
    "psoas_hematoma": "psoas hematoma",
}


class PatientInput(BaseModel):
    Sex: int
    Age: float
    BMI: float
    prior_back_surgeries: int = 0

    dx_adjacent_segment: int = 0
    dx_spondylolisthesis: int = 0
    dx_spondylosis: int = 0
    dx_stenosis: int = 0
    dx_scoliosis: int = 0
    dx_flat_back: int = 0
    dx_sagittal_imbalance: int = 0
    dx_post_laminectomy: int = 0
    dx_deformity: int = 0

    T12_L1: int = 0
    L1_L2: int = 0
    L2_L3: int = 0
    L3_L4: int = 0
    L4_L5: int = 0
    L5_S1: int = 0

    Open: int = 0
    Perc_screws: int = 0
    Lateral_Count: int = 0
    ALIF_Count: int = 0
    ACR: int = 0

    Average_PI: float
    PI_LL_mismatch: float
    ABS_PI_LL_mismatch: float
    PI_gt_50: int = 0

    post_LL: float
    post_SVA: float
    post_PI: float
    post_PT: float
    Post_op_SS: float

    LOS: float
    Open_Check_V2: int = 0
    Standalone_XLIF_Check: int = 0
    Retroperitoneal_Approach_LLIF_ALIF: int = 0
    Anterior_Posterior_Apporoach: int = 0
    Osteotomies_yes_no: int = 0

    infection_1_yes: int = 0
    DVT_1_yes: int = 0
    PE_1_yes: int = 0
    MI_1_yes: int = 0
    femoral_palsy_1_yes: int = 0
    hip_flexion_weakness_1_yes: int = 0
    acute_thigh_paresthesia: int = 0
    psoas_hematoma: int = 0


state: dict = {}


def _load_deepsurv(bundle_path: str = "models/deepsurv_bundle.pkl"):
    """Rehydrate the pycox CoxPH model from the saved bundle. Returns None
    if the bundle is missing — DeepSurv endpoint will then 503."""
    try:
        with open(bundle_path, "rb") as f:
            ds = pickle.load(f)
    except FileNotFoundError:
        print("WARN: deepsurv_bundle.pkl not found — /predict/asd/deepsurv disabled.")
        return None
    net = tt.practical.MLPVanilla(
        in_features=ds["in_features"],
        num_nodes=ds["num_nodes"],
        out_features=1,
        batch_norm=True,
        dropout=ds["dropout"],
        output_bias=False,
    )
    net.load_state_dict(ds["state_dict"])
    net.eval()
    model = CoxPH_NN(net, tt.optim.Adam())
    model.baseline_hazards_ = ds["baseline_hazards"]
    model.baseline_cumulative_hazards_ = ds["baseline_cumulative_hazards"]
    return {
        "model": model,
        "feature_cols": ds["feature_columns"],
        "encoded_columns": ds["encoded_columns"],
        "rare_cols": set(ds["rare_cols"]),
        "numeric_present": ds["numeric_present"],
        "scaler": ds["scaler"],
        "pca": ds["pca"],
        "cohort_risk": np.asarray(ds["cohort_risk"]).reshape(-1),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("models/rsf_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    state["rsf"] = bundle["model"]
    state["feature_cols"] = bundle["feature_columns"]
    state["encoded_columns"] = bundle["encoded_columns"]
    state["rare_cols"] = set(bundle["rare_cols"])
    state["numeric_present"] = bundle["numeric_present"]
    state["scaler"] = bundle["scaler"]
    state["pca"] = bundle["pca"]
    # Score the cohort once so /predict/asd doesn't run the whole forest over
    # 546 rows on every request.
    state["cohort_risk"] = state["rsf"].predict(bundle["X_vals"])
    state["pca_loadings"] = {
        f"pca_num_{i+1}": dict(zip(state["numeric_present"], state["pca"].components_[i]))
        for i in range(state["pca"].components_.shape[0])
    }
    # Fields the wire-format exposes that don't map to any model column — log
    # them once at startup so a typo doesn't silently degrade predictions.
    encoded = set(state["encoded_columns"])
    unmapped = [f for f, c in FIELD_TO_COL.items() if c not in encoded]
    if unmapped:
        print("WARN: PatientInput fields not present in encoded_columns:", unmapped)

    # DeepSurv (best calibration in CV — ECE 0.039 @ 12mo, 0.052 @ 24mo).
    state["deepsurv"] = _load_deepsurv()
    if state["deepsurv"] is not None:
        torch.set_num_threads(1)  # serving on a single Render worker, no need for more
        print("DeepSurv bundle loaded.")

    # SHAP explainer for RSF — model-agnostic PermutationExplainer because
    # sksurv's RSF is not natively supported by shap.TreeExplainer. Build
    # once at startup with a 50-sample background drawn from the training
    # cohort; per-patient explanation then takes ~5–12s on first call,
    # ~0.6s per call thereafter.
    try:
        rng = np.random.RandomState(42)
        X_train = np.asarray(bundle["X_vals"], dtype=float)
        bg_idx = rng.choice(len(X_train), 50, replace=False)
        state["shap_bg"] = X_train[bg_idx]
        state["shap_explainer"] = shap.PermutationExplainer(
            state["rsf"].predict, state["shap_bg"]
        )
        print("SHAP explainer built (background=50).")
    except Exception as ex:
        print(f"WARN: SHAP explainer failed to initialize: {ex}")
        state["shap_explainer"] = None
    yield


app = FastAPI(title="ASD Survival API", version="1.5", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": "rsf" in state,
        "deepsurv_loaded": state.get("deepsurv") is not None,
        "shap_loaded": state.get("shap_explainer") is not None,
    }


@app.get("/model-info")
def model_info():
    return {"pca_loadings": state["pca_loadings"]}


def build_raw_row(payload: dict) -> pd.DataFrame:
    # Build the row in the *encoded* column space (pre-PCA, pre-rare-drop).
    # The original app keyed off feature_cols, which had the numeric features
    # already replaced by pca_num_*, so the PCA scaler ran on zeros every call.
    row = {c: 0 for c in state["encoded_columns"]}
    for field, val in payload.items():
        col = FIELD_TO_COL.get(field)
        if col is not None and col in row:
            row[col] = val
    # Variant C: the model uses a single levels_fused_count integer rather
    # than the six per-level binaries the form sends. Compute it here from
    # the form's level toggles so the form layout doesn't have to change.
    if "levels_fused_count" in row:
        level_keys = ["T12_L1", "L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
        row["levels_fused_count"] = sum(int(payload.get(k, 0) or 0) for k in level_keys)
    return pd.DataFrame([row])


def preprocess(df_raw: pd.DataFrame):
    X = df_raw[state["encoded_columns"]]
    X = X.drop(columns=[c for c in state["rare_cols"] if c in X.columns], errors="ignore")

    numeric_present = state["numeric_present"]
    X_num = X[numeric_present].astype(float).values
    X_num_scaled = state["scaler"].transform(X_num)
    pcs = state["pca"].transform(X_num_scaled)

    df_pcs = pd.DataFrame(
        pcs,
        columns=[f"pca_num_{i+1}" for i in range(pcs.shape[1])],
        index=X.index,
    )
    X_final = pd.concat([X.drop(columns=numeric_present), df_pcs], axis=1)
    X_final = X_final[state["feature_cols"]]
    return X_final.values.astype(float), df_pcs.iloc[0].to_dict()


@app.post("/predict/asd")
def predict_asd(inp: PatientInput):
    try:
        payload = inp.model_dump()
        X_raw = build_raw_row(payload)
        Xp, pca_scores = preprocess(X_raw)

        rsf = state["rsf"]
        surv_fn = rsf.predict_survival_function(Xp, return_array=False)[0]
        times = surv_fn.x.tolist()
        probs = surv_fn.y.tolist()

        def interp(t: float) -> float:
            return float(np.interp(t, surv_fn.x, surv_fn.y))

        horizons = {t: interp(t) for t in [12, 24, 36, 60]}

        median = None
        for t, s in zip(surv_fn.x, surv_fn.y):
            if s <= 0.5:
                median = float(t)
                break

        patient_risk = float(rsf.predict(Xp)[0])
        cohort_risk = state["cohort_risk"]
        percentile = float((cohort_risk < patient_risk).mean() * 100)

        return {
            "survival_curve": {"times": times, "probs": probs},
            "asd_free_prob": horizons,
            "median_time_months": median,
            "risk_percentile": percentile,
            "patient_risk": patient_risk,
            "pca_scores": pca_scores,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _preprocess_for_deepsurv(df_raw: pd.DataFrame):
    """DeepSurv has its own preprocessing components (same shape as RSF's,
    but kept separate so the two models can evolve independently)."""
    ds = state["deepsurv"]
    X = df_raw[ds["encoded_columns"]]
    X = X.drop(columns=[c for c in ds["rare_cols"] if c in X.columns], errors="ignore")
    X_num = X[ds["numeric_present"]].astype(float).values
    pcs = ds["pca"].transform(ds["scaler"].transform(X_num))
    df_pcs = pd.DataFrame(
        pcs,
        columns=[f"pca_num_{i+1}" for i in range(pcs.shape[1])],
        index=X.index,
    )
    X_final = pd.concat([X.drop(columns=ds["numeric_present"]), df_pcs], axis=1)
    X_final = X_final[ds["feature_cols"]]
    return X_final.values.astype("float32"), df_pcs.iloc[0].to_dict()


@app.post("/predict/asd/deepsurv")
def predict_asd_deepsurv(inp: PatientInput):
    """DeepSurv prediction — same response shape as /predict/asd. Best
    calibration in cross-validation (ECE 0.039 @ 12mo, 0.052 @ 24mo);
    use this endpoint for pre-op patient counseling."""
    if state.get("deepsurv") is None:
        raise HTTPException(status_code=503, detail="DeepSurv bundle not loaded")
    try:
        ds = state["deepsurv"]
        payload = inp.model_dump()
        X_raw = build_raw_row(payload)
        Xp, pca_scores = _preprocess_for_deepsurv(X_raw)

        # Survival function -> aligned (times, probs)
        with torch.no_grad():
            surv_df = ds["model"].predict_surv_df(Xp)
        times = surv_df.index.values.astype(float)
        probs = surv_df.iloc[:, 0].values.astype(float)

        def interp(t: float) -> float:
            return float(np.interp(t, times, probs))

        horizons = {t: interp(t) for t in [12, 24, 36, 60]}

        median = None
        for t, s in zip(times, probs):
            if s <= 0.5:
                median = float(t)
                break

        # Risk score: pycox returns log-partial-hazard; compare against the
        # cohort vector saved at training time (DeepSurv-specific scale).
        with torch.no_grad():
            patient_risk = float(ds["model"].predict(Xp).reshape(-1)[0])
        cohort_risk = ds["cohort_risk"]
        percentile = float((cohort_risk < patient_risk).mean() * 100)

        return {
            "survival_curve": {"times": times.tolist(), "probs": probs.tolist()},
            "asd_free_prob": horizons,
            "median_time_months": median,
            "risk_percentile": percentile,
            "patient_risk": patient_risk,
            "pca_scores": pca_scores,
            "model": "deepsurv",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/asd/ensemble")
def predict_asd_ensemble(inp: PatientInput):
    """RSF + DeepSurv ensemble. Best discrimination in cross-validation
    (C = 0.597 vs RSF 0.591 / DeepSurv 0.578); slightly worse calibration
    than DeepSurv alone but more reliable on borderline cases. Survival
    functions are averaged on a common time grid; percentiles are the mean
    of each component model's cohort percentile."""
    if state.get("deepsurv") is None:
        raise HTTPException(status_code=503, detail="DeepSurv bundle not loaded")
    try:
        ds = state["deepsurv"]
        payload = inp.model_dump()
        X_raw = build_raw_row(payload)

        # RSF side
        Xp_rsf, pca_scores = preprocess(X_raw)
        rsf = state["rsf"]
        risk_rsf = float(rsf.predict(Xp_rsf)[0])
        rsf_surv = rsf.predict_survival_function(Xp_rsf, return_array=False)[0]

        # DeepSurv side
        Xp_ds, _ = _preprocess_for_deepsurv(X_raw)
        with torch.no_grad():
            ds_surv_df = ds["model"].predict_surv_df(Xp_ds)
            risk_ds = float(ds["model"].predict(Xp_ds).reshape(-1)[0])

        # Common monthly grid covering the longer of the two curves
        max_t = max(float(rsf_surv.x[-1]), float(ds_surv_df.index.values[-1]))
        time_grid = np.arange(0, int(np.ceil(max_t)) + 1, dtype=float)

        s_rsf = np.interp(time_grid, rsf_surv.x, rsf_surv.y,
                          left=1.0, right=float(rsf_surv.y[-1]))
        ds_t = ds_surv_df.index.values.astype(float)
        ds_p = ds_surv_df.iloc[:, 0].values.astype(float)
        s_ds = np.interp(time_grid, ds_t, ds_p, left=1.0, right=float(ds_p[-1]))

        s_avg = (s_rsf + s_ds) / 2.0

        def interp(t: float) -> float:
            return float(np.interp(t, time_grid, s_avg))

        horizons = {t: interp(t) for t in [12, 24, 36, 60]}

        median = None
        for t, s in zip(time_grid, s_avg):
            if s <= 0.5:
                median = float(t)
                break

        # Percentile = mean of each model's cohort percentile (rank-average).
        # The two risk-score scales differ (RSF: cumulative hazard sum,
        # DeepSurv: log-partial-hazard), so we can't average raw risks —
        # we average where each patient sits in *its own* cohort.
        rsf_pct = float((state["cohort_risk"] < risk_rsf).mean() * 100)
        ds_pct = float((ds["cohort_risk"] < risk_ds).mean() * 100)
        percentile = (rsf_pct + ds_pct) / 2.0

        return {
            "survival_curve": {"times": time_grid.tolist(), "probs": s_avg.tolist()},
            "asd_free_prob": horizons,
            "median_time_months": median,
            "risk_percentile": percentile,
            "pca_scores": pca_scores,
            "model": "ensemble",
            "components": {
                "rsf_percentile": rsf_pct,
                "deepsurv_percentile": ds_pct,
                "rsf_5y": float(np.interp(60, rsf_surv.x, rsf_surv.y)),
                "deepsurv_5y": float(np.interp(60, ds_t, ds_p)),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# SHAP explanation — streaming endpoint (Server-Sent Events)
# ----------------------------------------------------------------------

def _pretty_feature(name: str) -> str:
    if name == "pca_num_1": return "Post-op alignment composite"
    if name == "pca_num_2": return "Operative burden composite"
    if name == "pca_num_3": return "Sagittal mismatch composite"
    if name == "(1 = PI>50)": return "Pelvic incidence > 50°"
    if name == "Anterior + Posterior Apporoach": return "Anterior + posterior approach"
    if name == "prior back surgeries? (y=1)": return "Prior back surgery"
    if name == "Perc screws?": return "Percutaneous screws"
    if name == "Standalone XLIF Check": return "Standalone XLIF"
    if name == "Open Check V2": return "Open approach (V2)"
    if name == "Retroperitoneal Approach (LLIF ± ALIF)": return "Retroperitoneal LLIF/ALIF"
    if name == "Osteotomies (yes/no)": return "Osteotomies"
    if name == "ACR (y=1)": return "Anterior column release"
    if name.startswith("dx_"): return "Diagnosis: " + name[3:].replace("_", " ")
    if name == "length of hospital stay (d)": return "Length of stay"
    return name


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/predict/asd/explain")
async def explain_asd(inp: PatientInput):
    """Per-patient SHAP attribution for the RSF prediction.

    Returns Server-Sent Events because permutation SHAP can take 5–12s on
    the first call and ~0.6s thereafter. Events:
      starting   — preprocessing the input
      computing  — SHAP in progress, with elapsed seconds
      ready      — final payload with top contributors
      error      — anything went wrong
    """
    if state.get("shap_explainer") is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded")

    payload = inp.model_dump()

    async def generate():
        try:
            yield _sse({"status": "starting",
                        "message": "Preparing patient features…"})
            X_raw = build_raw_row(payload)
            Xp, _ = preprocess(X_raw)
            yield _sse({"status": "computing", "elapsed": 0,
                        "message": "Computing SHAP attributions…"})

            # Run SHAP in a thread so the event loop can keep emitting
            # keep-alive events.
            task = asyncio.create_task(asyncio.to_thread(
                state["shap_explainer"], Xp
            ))
            elapsed = 0
            while not task.done():
                await asyncio.sleep(1.0)
                elapsed += 1
                yield _sse({"status": "computing", "elapsed": elapsed,
                            "message": f"Computing SHAP… ({elapsed}s)"})
            shap_obj = await task

            sv = np.asarray(shap_obj.values[0], dtype=float)
            base = float(shap_obj.base_values[0])

            feats = state["feature_cols"]
            patient_x = Xp[0]
            order = np.argsort(np.abs(sv))[::-1][:10]
            contributors = [
                {
                    "feature": feats[i],
                    "display_name": _pretty_feature(feats[i]),
                    "shap_value": float(sv[i]),
                    "patient_value": float(patient_x[i]),
                    "direction": "increases" if sv[i] > 0 else "decreases",
                }
                for i in order
            ]

            patient_risk = float(state["rsf"].predict(Xp)[0])
            cohort_risk = state["cohort_risk"]
            percentile = float((cohort_risk < patient_risk).mean() * 100)

            yield _sse({
                "status": "ready",
                "elapsed": elapsed,
                "contributors": contributors,
                "base_value": base,
                "patient_risk": patient_risk,
                "risk_percentile": percentile,
                "model": "rsf",
                "note": (
                    "SHAP attributions reflect patterns the RSF learned from the "
                    "training cohort, which may not fully match published causal "
                    "risk relationships — particularly for patients with extensive "
                    "deformity (cohort selection effects). Interpret with clinical "
                    "judgment."
                ),
            })
        except Exception as e:
            yield _sse({"status": "error", "message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache, no-transform",
            "connection": "keep-alive",
            "x-accel-buffering": "no",   # disables nginx-style buffering on Render
        },
    )
