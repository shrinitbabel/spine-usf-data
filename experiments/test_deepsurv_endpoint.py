"""Hit /predict/asd/deepsurv with the 8 frontend presets and print results."""
from fastapi.testclient import TestClient
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

PRESETS = {
    "Low risk (short fusion)": dict(
        Sex=1, Age=50, BMI=24, prior_back_surgeries=0,
        dx_stenosis=1, L4_L5=1, L5_S1=0,
        Open=0, Perc_screws=1,
        Standalone_XLIF_Check=1, Retroperitoneal_Approach_LLIF_ALIF=1,
        Lateral_Count=0, ALIF_Count=0, ACR=0,
        Average_PI=48, PI_LL_mismatch=4, ABS_PI_LL_mismatch=4, PI_gt_50=0,
        post_LL=50, post_SVA=10, post_PI=48, post_PT=13, Post_op_SS=35,
        LOS=2,
    ),
    "Adjacent stress": dict(
        Sex=1, Age=63, BMI=28, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=0,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=58, PI_LL_mismatch=15, ABS_PI_LL_mismatch=15, PI_gt_50=1,
        post_LL=45, post_SVA=35, post_PI=58, post_PT=22, Post_op_SS=28,
        LOS=4,
    ),
    "Deformity / long construct": dict(
        Sex=1, Age=75, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_scoliosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies_yes_no=1, Anterior_Posterior_Apporoach=1, ACR=1,
        Lateral_Count=3, ALIF_Count=0,
        Average_PI=65, PI_LL_mismatch=25, ABS_PI_LL_mismatch=25, PI_gt_50=1,
        post_LL=40, post_SVA=65, post_PI=65, post_PT=30, Post_op_SS=20,
        LOS=8,
    ),
    "Single-level spondy": dict(
        Sex=1, Age=60, BMI=27, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylolisthesis=1,
        L4_L5=1, L5_S1=0,
        Open=1, Lateral_Count=1, ALIF_Count=0,
        Average_PI=52, PI_LL_mismatch=10, ABS_PI_LL_mismatch=10, PI_gt_50=1,
        post_LL=45, post_SVA=25, post_PI=52, post_PT=16, Post_op_SS=30,
        LOS=3,
    ),
    "Revision lumbar": dict(
        Sex=1, Age=68, BMI=30, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1,
        L4_L5=1, L5_S1=1,
        Open=1, Perc_screws=1, Osteotomies_yes_no=1, Anterior_Posterior_Apporoach=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=56, PI_LL_mismatch=18, ABS_PI_LL_mismatch=18, PI_gt_50=1,
        post_LL=45, post_SVA=40, post_PI=56, post_PT=24, Post_op_SS=27,
        LOS=5,
    ),
    "Flat-back correction": dict(
        Sex=1, Age=71, BMI=29, prior_back_surgeries=1,
        dx_stenosis=1, dx_flat_back=1, dx_sagittal_imbalance=1,
        T12_L1=1, L1_L2=1, L2_L3=1, L3_L4=1, L4_L5=1, L5_S1=1,
        Open=1, Osteotomies_yes_no=1, Anterior_Posterior_Apporoach=1, ACR=1,
        Lateral_Count=2, ALIF_Count=0,
        Average_PI=60, PI_LL_mismatch=25, ABS_PI_LL_mismatch=25, PI_gt_50=1,
        post_LL=42, post_SVA=55, post_PI=60, post_PT=28, Post_op_SS=22,
        LOS=8,
    ),
    "MIS short fusion": dict(
        Sex=1, Age=55, BMI=26, prior_back_surgeries=0,
        dx_stenosis=1, L4_L5=1,
        Open=0, Perc_screws=1,
        Standalone_XLIF_Check=1, Retroperitoneal_Approach_LLIF_ALIF=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=50, PI_LL_mismatch=6, ABS_PI_LL_mismatch=6, PI_gt_50=1,
        post_LL=48, post_SVA=18, post_PI=50, post_PT=14, Post_op_SS=33,
        LOS=2,
    ),
    "Elderly mismatch": dict(
        Sex=1, Age=78, BMI=32, prior_back_surgeries=1,
        dx_stenosis=1, dx_spondylosis=1, dx_sagittal_imbalance=1,
        L4_L5=1, L5_S1=1,
        Open=1, Osteotomies_yes_no=1,
        Lateral_Count=1, ALIF_Count=0,
        Average_PI=62, PI_LL_mismatch=22, ABS_PI_LL_mismatch=22, PI_gt_50=1,
        post_LL=45, post_SVA=50, post_PI=62, post_PT=26, Post_op_SS=25,
        LOS=6,
    ),
}

with TestClient(app) as client:
    print(f"\nVersion: {client.get('/openapi.json').json()['info']['version']}")
    print(f"Health: {client.get('/healthz').json()}\n")

    rows_rsf, rows_ds = [], []
    for name, p in PRESETS.items():
        for endpoint, rows in [("/predict/asd", rows_rsf),
                               ("/predict/asd/deepsurv", rows_ds)]:
            r = client.post(endpoint, json=p)
            if r.status_code != 200:
                print(f"  {name} {endpoint} -> {r.status_code} {r.text[:80]}")
                continue
            d = r.json()
            rows.append({
                "preset": name,
                "pct": d["risk_percentile"],
                "p1y": d["asd_free_prob"]["12"],
                "p2y": d["asd_free_prob"]["24"],
                "p3y": d["asd_free_prob"]["36"],
                "p5y": d["asd_free_prob"]["60"],
                "median": d["median_time_months"],
            })

    def show(rows, label):
        print(f"=== {label} ===")
        print(f"  {'Preset':32s}  pct   1y     2y     3y     5y     median")
        for r in sorted(rows, key=lambda x: x["pct"]):
            med = f"{r['median']:.0f}mo" if r['median'] else "  >FU"
            print(f"  {r['preset']:32s}  {r['pct']:>4.0f}  "
                  f"{r['p1y']*100:5.1f}  {r['p2y']*100:5.1f}  "
                  f"{r['p3y']*100:5.1f}  {r['p5y']*100:5.1f}  {med:>6s}")
        if rows:
            spread = max(r['p5y'] for r in rows) - min(r['p5y'] for r in rows)
            print(f"  -> 5y spread: {spread*100:.1f} pts\n")

    show(rows_rsf, "RSF (current production)")
    show(rows_ds, "DeepSurv (new endpoint)")
