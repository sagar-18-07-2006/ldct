"""app_site.py - the LDCT-Recon project website (Streamlit).

    streamlit run app_site.py

A multi-section project site: overview, background, data and scope, method, results, live demo, limitations and
references.  It does NOT load PyTorch: every number and figure on the Results pages is read from the files your own
runs wrote (results/<organ>/...), so the site always shows what was really measured.  The site is scoped to abdominal CT.

The Live demo page talks to the inference server (server.py) - start it first if you want to use that page.
Research prototype - not for clinical use.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------------------- edit these
PROJECT = dict(
    name="LDCT-Recon",
    tagline="Deep-learning enhancement of low-dose CT scans",
    author="Sagar Maheshwari",
    course="Digital Image Processing",   # e.g. "Digital Image Processing, Semester 5"
    guide="",                            # your professor's name (leave "" to hide)
    institution="",                      # your college (leave "" to hide)
    year="2026",
)
ROOT = Path(os.environ.get("LDCT_HOME", Path(__file__).resolve().parent))  # folder that contains data/, runs/, results/
ORGAN_TITLES = {"liver": "Abdomen (liver)", "chest": "Chest"}
ARCH_LABELS = {"cnn": "CNN (3-layer)", "dncnn": "DnCNN", "redcnn": "RED-CNN", "unet": "U-Net", "resunet": "Residual U-Net",
               "restormer": "Restormer-lite"}

st.set_page_config(page_title=f"{PROJECT['name']} - low-dose CT enhancement", page_icon="🩻", layout="wide")

st.markdown("""
<style>
 .block-container {padding-top: 1.2rem; max-width: 1180px;}
 .hero {padding: 2rem 2.2rem; border-radius: 16px; margin-bottom: 1.4rem; color: #fff;
        background: linear-gradient(120deg, #0b2540 0%, #14506b 60%, #1f7a8c 100%);}
 .hero h1 {margin: 0 0 .3rem 0; font-size: 2.3rem; color: #fff; letter-spacing: -.5px;}
 .hero p {margin: .15rem 0; font-size: 1.08rem; opacity: .92;}
 .hero .meta {font-size: .88rem; opacity: .75; margin-top: .9rem;}
 .tile {min-height: 8.4rem; padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);}
 .tile .num {font-size: 2rem; font-weight: 700; line-height: 1.1; color: #1f9bb5;}
 .tile .lab {font-size: .86rem; opacity: .85; margin-top: .25rem;}
 .card {padding: 1rem 1.2rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.06); height: 100%;}
 .card h4 {margin: 0 0 .4rem 0;}
 .pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px;}
 .ok {background:#d9f2e3; color:#145a32;} .warn {background:#fdeccb; color:#7d4e00;} .no {background:#f8d9d9; color:#7b1d1d;}
 .note {padding: .8rem 1rem; border-left: 4px solid #1f9bb5; background: rgba(31,155,181,.09); border-radius: 6px; margin: .6rem 0;}
 .caution {padding: .8rem 1rem; border-left: 4px solid #d98e04; background: rgba(217,142,4,.10); border-radius: 6px; margin: .6rem 0;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ helpers
def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def res_dir(organ: str) -> Path:
    return ROOT / "results" / organ


def organ_title(o: str) -> str:
    return ORGAN_TITLES.get(o, o.title())


def organs_with_results() -> list[str]:
    d = ROOT / "results"
    return sorted(p.name for p in d.iterdir() if (p / "comparison.json").exists()) if d.exists() else []


def n_cached_patients(organ: str) -> int:
    d = ROOT / "data" / "cache" / organ
    if not d.exists():
        return 0
    pids = set()
    for m in d.glob("*/meta.json"):
        j = read_json(m, {})
        pids.add(j.get("patient_id", m.parent.name))
    return len(pids)


def trained_archs(organ: str) -> list[str]:
    d = ROOT / "runs" / organ
    return sorted(p.name for p in d.iterdir() if (p / "best.pt").exists()) if d.exists() else []


def table(df: pd.DataFrame) -> None:
    """Static table whose long text wraps (st.dataframe clips it)."""
    try:
        st.table(df.style.hide(axis="index"))
    except Exception:
        st.dataframe(df, hide_index=True)


def tile(num: str, label: str) -> str:
    return f'<div class="tile"><div class="num">{num}</div><div class="lab">{label}</div></div>'


def card(title: str, body: str) -> str:
    return f'<div class="card"><h4>{title}</h4>{body}</div>'


def pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{text}</span>'


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not found: {path.relative_to(ROOT) if ROOT in path.parents else path}. Run the step that creates it.")


def headline(organ: str) -> dict | None:
    """The numbers on the overview tiles, read from the real result files."""
    cv = read_json(res_dir(organ) / "cv" / "cv_summary.json")
    base = read_json(res_dir(organ) / "extras" / "baselines.json")
    out: dict = {}
    if cv:
        out.update(src="cross-validation", n=cv["psnr"]["n"], psnr_gain=cv["psnr"]["mean_gain"],
                   rmse_in=cv["rmse"]["input_mean"], rmse_out=cv["rmse"]["output_mean"], improved=cv["psnr"]["improved"])
    else:
        rows = read_json(res_dir(organ) / "comparison.json")
        if not rows:
            return None
        best = max(rows, key=lambda r: r["val_psnr"])
        out.update(src="test set", n=None, psnr_gain=best["psnr_out"] - best["psnr_in"], rmse_in=best["rmse_in"],
                   rmse_out=best["rmse_out"], improved=None)
    if base:
        deep = next((v for k, v in base.items() if "deep" in k.lower()), None)
        classical = [v for k, v in base.items() if "deep" not in k.lower() and not k.lower().startswith("low-dose")]
        if deep and classical:
            out["vs_classical_db"] = deep["psnr"] - max(c["psnr"] for c in classical)
    return out


# ---------------------------------------------------------------------------------------------------- pages
def page_overview() -> None:
    meta = " | ".join(x for x in [PROJECT["author"], PROJECT["course"], PROJECT["guide"], PROJECT["institution"], PROJECT["year"]] if x)
    st.markdown(f"""<div class="hero"><h1>{PROJECT['name']}</h1><p>{PROJECT['tagline']}</p>
<p>A neural network that removes noise from low-radiation CT slices while keeping anatomy intact, evaluated
on real patient data against classical image-processing filters.</p><div class="meta">{meta}</div></div>""", unsafe_allow_html=True)

    organs = organs_with_results()
    h = headline(organs[0]) if organs else None
    if h:
        c = st.columns(4)
        c[0].markdown(tile(f"{h['psnr_gain']:+.1f} dB", f"PSNR gain over the raw low-dose scan ({organ_title(organs[0])}, {h['src']})"), unsafe_allow_html=True)
        c[1].markdown(tile(f"{100 * (h['rmse_out'] / h['rmse_in'] - 1):+.0f}%", f"error in HU ({h['rmse_in']:.1f} to {h['rmse_out']:.1f})"), unsafe_allow_html=True)
        c[2].markdown(tile(f"{h['improved']} / {h['n']}" if h["improved"] is not None else "n/a",
                           "patients improved, each tested by a model that never saw them" if h["improved"] is not None else "run cv.py for per-patient results"),
                      unsafe_allow_html=True)
        c[3].markdown(tile(f"{h['vs_classical_db']:+.1f} dB" if "vs_classical_db" in h else "n/a",
                           "over the best tuned classical filter (non-local means)" if "vs_classical_db" in h else "run extras.py for classical baselines"),
                      unsafe_allow_html=True)
    else:
        st.info("No results found yet. Train a model and run `python model.py compare`, `python extras.py all` and `python cv.py` to fill this page.")

    st.markdown("### What this project does")
    a, b, c = st.columns(3)
    a.markdown(card("The problem", "<p>CT uses X-rays. Lowering the dose protects patients, but fewer photons reach the detector, "
                    "so the image becomes grainy and small lesions are harder to see. Noise grows roughly as 1/sqrt(dose).</p>"), unsafe_allow_html=True)
    b.markdown(card("The approach", "<p>Using real scans that exist at two dose levels, a neural network learns to map the noisy slice to the routine-dose slice. "
                    "It estimates the noise and subtracts it, using the neighbouring slices as context.</p>"), unsafe_allow_html=True)
    c.markdown(card("The evidence", "<p>Five network designs and three classical filters were compared on patients held out from training, "
                    "with patient-level cross-validation, PSNR, SSIM and error in Hounsfield units.</p>"), unsafe_allow_html=True)

    st.markdown("### How the pieces fit")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent";
 node [shape=box, style="rounded,filled", fillcolor="#dcecf4", color="#1f7a8c", fontname="Helvetica", fontsize=11, fontcolor="#0b2540"];
 edge [color="#5b7c8d"];
 A [label="Mayo LDCT data\\n(TCIA / IDC)"]; B [label="DICOM to HU\\nwindow, pair slices"]; C [label="Patient-level\\nsplit / folds"];
 D [label="Train 5 networks\\n(2.5D input)"]; E [label="Evaluate vs filters\\nPSNR SSIM RMSE"]; F [label="Inference server\\n(FastAPI + GPU)"]; G [label="Web client\\n(this site)"];
 A -> B -> C -> D -> E; D -> F -> G; }""")

    st.markdown("### Scope at a glance")
    rows = []
    for o in ["liver"]:
        have = o in organs
        rows.append({"Organ": organ_title(o), "Patients cached": n_cached_patients(o), "Models trained": len(trained_archs(o)),
                     "Evaluated": "yes" if have else "not yet"})
    table(pd.DataFrame(rows))
    st.caption("Everything shown on this site was measured in this project on abdominal CT. Nothing is estimated or copied from other papers.")


def page_background() -> None:
    st.title("Background")
    st.markdown("### Why low-dose CT is noisy")
    st.markdown("""
* A CT scanner measures how many X-ray photons survive a path through the body. Photon counting is a **Poisson process**, so the
  random error scales as 1 / sqrt(dose): a quarter of the dose gives about **twice the noise**.
* The image is rebuilt from projections by **filtered back-projection (FBP)**. FBP boosts high frequencies, so the noise is
  *correlated*: it has a streaky texture and is not like TV static. The noise power spectrum on the Results page shows this.
* Noise matters because it hides **low-contrast structures**, such as small liver lesions and faint vessels.""")
    st.latex(r"\sigma \propto \frac{1}{\sqrt{D}}")
    st.markdown("### Why not simply blur it?")
    st.markdown("""
Gaussian and median filters smooth noise and edges alike. Non-local means is smarter (it averages similar patches) and is
used here as a **strong classical baseline**. A learned model can use what noise looks like across many patients to remove it
more selectively. The Results page measures how much better it really is.""")
    st.markdown("### Key terms")
    table(pd.DataFrame([
        ("Hounsfield unit (HU)", "The CT brightness scale: water is 0, air is -1000, soft tissue is tens of HU."),
        ("Window", "The HU range shown as black to white. The abdomen uses about -160 to 240 HU."),
        ("PSNR (dB)", "Signal strength compared with the leftover error. Higher is better; +3 dB halves the error power."),
        ("SSIM (0 to 1)", "Similarity of local structure and contrast. 1 means identical."),
        ("RMSE (HU)", "Typical per-pixel error in Hounsfield units. Lower is better."),
        ("2.5D input", "Three neighbouring slices go in, one cleaned slice comes out."),
    ], columns=["Term", "Meaning"]))


def page_data() -> None:
    st.title("Data and scope")
    st.markdown("""
The data is the **Mayo Clinic "Low Dose CT Image and Projection Data"** collection on The Cancer Imaging Archive
(DOI 10.7937/9npb-2637, licence CC BY 4.0). For each patient it holds the **same scan at two dose levels**: a routine-dose
(reference) series and a low-dose series made by inserting realistic noise into the raw projection data.""")
    table(pd.DataFrame([
        ("Collection", "Chest (about 100 patients), abdomen / liver (about 100) and head (restricted access); 299 patients plus a phantom"),
        ("Low-dose level", "25% of routine dose for abdomen, 10% for chest (simulated)"),
        ("Scanners", "Siemens and GE. Only the Siemens cases provide both a Full Dose and a Low Dose image series"),
        ("What was used here", "Abdomen: 15 Siemens patients, 5 mm slices, from the Imaging Data Commons"),
        ("Split", "By patient. Quick test: 11 train / 2 validation / 2 test. Final evidence: 5-fold patient-level cross-validation"),
    ], columns=["Item", "Detail"]))
    st.markdown('<div class="note"><b>Why a patient-level split?</b> Neighbouring slices of one person are almost identical. '
                'Putting some of them in training and some in testing would leak the answer and inflate every score.</div>', unsafe_allow_html=True)

    st.markdown("### Scope")
    organs = organs_with_results()
    table(pd.DataFrame([{"Region": "Abdomen (liver)", "Protocol": "Abdominal CT, 25% dose, soft-tissue window",
                         "Patients cached": n_cached_patients("liver"), "Models trained": len(trained_archs("liver")),
                         "Results": "available" if "liver" in organs else "none yet"}]))
    st.markdown('<div class="note"><b>Scope.</b> This project focuses on <b>abdominal CT</b> and goes deep on it: five network designs, '
                'tuned classical baselines and patient-level cross-validation. The Mayo collection also contains chest and head scans; '
                'they are outside the scope of this project and are not evaluated.</div>', unsafe_allow_html=True)


def page_method() -> None:
    st.title("Method")
    st.markdown("### Pipeline")
    table(pd.DataFrame([
        ("1", "Convert", "DICOM to Hounsfield units using slope and intercept; pair each low-dose slice with its routine-dose slice."),
        ("2", "Window", "Clip to the organ's HU window and scale to [0, 1]."),
        ("3", "Split", "By patient (never by slice)."),
        ("4", "Sample", "Random 128 x 128 patches of three stacked neighbouring slices (2.5D); patches outside the body are skipped."),
        ("5", "Train", "Composite loss, AdamW, warm-up then cosine learning rate, bf16 mixed precision, 30 epochs."),
        ("6", "Evaluate", "Full test slices: PSNR, SSIM, RMSE in HU, against tuned classical filters."),
    ], columns=["Step", "Stage", "What happens"]))

    st.markdown("### The five networks compared")
    table(pd.DataFrame([
        ("CNN (3-layer)", "Three convolutions, input straight to output. The simple baseline.", "Dong et al. 2016"),
        ("DnCNN", "17 convolution layers with batch norm; predicts the noise.", "Zhang et al. 2017"),
        ("RED-CNN", "5 convolution + 5 deconvolution layers with residual shortcuts.", "Chen et al. 2017"),
        ("U-Net", "Looks at the image at several scales; skip connections carry fine detail.", "Ronneberger et al. 2015"),
        ("Residual U-Net", "U-Net with residual blocks, GroupNorm, SiLU, strided down-sampling, bilinear up-sampling and a zero-initialised output layer.", "this project's design"),
    ], columns=["Model", "Idea", "Reference"]))
    st.markdown('<div class="note">All of them except the plain CNN output <b>the centre slice plus a learned correction</b>. '
                'The Residual U-Net starts as "change nothing" and learns only what to remove, which is why it rarely invents structure.</div>',
                unsafe_allow_html=True)

    st.markdown("### Training objective")
    st.latex(r"\mathcal{L}=\lVert\hat y-y\rVert_1+\lambda_e\lVert\nabla\hat y-\nabla y\rVert_1+\lambda_s\bigl(1-\mathrm{SSIM}(\hat y,y)\bigr)")
    st.markdown("""
* **L1** is the main accuracy term and blurs less than squared error.
* **SSIM** protects local structure and contrast.
* **Gradient (edge) term** keeps boundaries such as organ edges and vessel walls sharp.
* Weights are set per organ (abdomen: 1.0 / 0.2 / 0.1). They were set by hand, not tuned.""")

    st.markdown("### Evaluation protocol")
    st.markdown("""
* Metrics are computed against the **routine-dose scan** on slices from patients the model never saw.
* Classical filters (Gaussian, median, non-local means) are **tuned on the validation patients**, then applied unchanged to the test patients,
  so the comparison is fair.
* Statistics are computed **per patient** in the cross-validation, because slices of one patient are correlated.""")


def page_results() -> None:
    st.title("Results")
    organs = organs_with_results()
    if not organs:
        st.warning("No results found. Run `python model.py compare --organ liver ...` first.")
        return
    organ = organs[0] if len(organs) == 1 else st.radio("Organ", organs, format_func=organ_title, horizontal=True)
    d = res_dir(organ)
    st.caption("Results for abdominal CT.") if len(organs) == 1 else None
    t1, t2, t3, t4 = st.tabs(["Cross-validation", "Architectures", "Versus classical filters", "What the model removes"])

    with t1:
        cv = read_json(d / "cv" / "cv_summary.json")
        if not cv:
            st.info("Run `python cv.py --organ " + organ + "` to produce the cross-validation results.")
        else:
            n = cv["psnr"]["n"]
            st.markdown(f"Each of the **{n} patients** was tested by a model trained **without** that patient (5-fold, patient-level).")
            rows = []
            for k, lab, f in [("psnr", "PSNR (dB)", "{:.2f}"), ("ssim", "SSIM", "{:.4f}"), ("rmse", "RMSE (HU)", "{:.1f}")]:
                v = cv[k]
                rows.append({"Metric": lab, "Low-dose input": f"{f.format(v['input_mean'])} ± {f.format(v['input_std'])}",
                             "Model output": f"{f.format(v['output_mean'])} ± {f.format(v['output_std'])}",
                             "Patients improved": f"{v['improved']} / {v['n']}"})
            table(pd.DataFrame(rows))
            p = cv["psnr"]
            allbetter = p["improved"] == p["n"]
            st.caption(f"± is the spread between patients. {p['improved']} of {p['n']} patients improved (Wilcoxon p = {p['wilcoxon_p']:.1e}). "
                       + (f"With {p['n']} patients {2 / 2 ** p['n']:.1e} is the smallest p-value this test can give, so here it only means that every patient improved."
                          if allbetter else "Check the figure for the patients where it did not."))
            show_image(d / "cv" / "cv_per_patient.png", "Each patient: red = low-dose input, teal = model output; the grey line is the change.")
            st.markdown('<div class="note"><b>How to read it.</b> The length of each grey line is the change for that patient. '
                        'Look for short or reversed lines: those are the patients the model helps least.</div>', unsafe_allow_html=True)

    with t2:
        rows = read_json(d / "comparison.json")
        sig = read_json(d / "significance.json", {})
        best = read_json(d / "best_arch.json", {})
        if not rows:
            st.info("Run `python model.py compare` first.")
        else:
            base = rows[0]
            tbl = [{"Model": "Low-dose input", "PSNR (dB)": f"{base['psnr_in']:.2f}", "SSIM": f"{base['ssim_in']:.4f}",
                      "RMSE (HU)": f"{base['rmse_in']:.1f}", "ms / slice": "n/a", "Train (min)": "n/a", "Params (M)": "n/a"}]
            for r in rows:
                star = " (best)" if r["arch"] == best.get("arch") else ""
                tbl.append({"Model": r["label"] + star, "PSNR (dB)": f"{r['psnr_out']:.2f} ± {r['psnr_out_std']:.2f}",
                              "SSIM": f"{r['ssim_out']:.4f}", "RMSE (HU)": f"{r['rmse_out']:.1f}", "ms / slice": f"{r['infer_ms']:.0f}",
                              "Train (min)": f"{r['train_min']:.0f}", "Params (M)": f"{r['params_m']:.2f}"})
            table(pd.DataFrame(tbl))
            st.caption("Best = highest validation PSNR. The ± is the spread over test slices (different anatomy), not uncertainty about the mean.")
            show_image(d / "comparison_bars.png", "Dashed line = low-dose input. Note the axes do not start at zero, which exaggerates small gaps.")
            show_image(d / "comparison_visual.png", "Same slice through every model: output, zoomed region and absolute error against the routine-dose scan.")
            ranked = sorted(rows, key=lambda r: -r["psnr_out"])
            gap = ranked[0]["psnr_out"] - ranked[1]["psnr_out"] if len(ranked) > 1 else 0.0
            spread = ranked[0]["psnr_out"] - ranked[-1]["psnr_out"] if len(ranked) > 1 else 0.0
            st.markdown(f'<div class="caution"><b>Honest reading.</b> By test PSNR the top model is {ranked[0]["label"]}, {gap:.2f} dB ahead of the runner-up '
                        f'and {spread:.2f} dB ahead of the weakest. Gaps this size are easy to move with a different split or seed. The p-values in '
                        '<code>significance.json</code> treat slices as independent and come from few patients, so they are overstated. '
                        'The cross-validation tab is the stronger evidence.</div>', unsafe_allow_html=True)

    with t3:
        base = read_json(d / "extras" / "baselines.json")
        if not base:
            st.info("Run `python extras.py baselines --organ " + organ + "`.")
        else:
            rows = []
            for k, v in base.items():
                rows.append({"Method": k, "PSNR (dB)": f"{v['psnr']:.2f}", "SSIM": f"{v['ssim']:.4f}", "RMSE (HU)": f"{v['rmse_hu']:.1f}"})
            table(pd.DataFrame(rows))
            n = next(iter(base.values())).get("n_slices")
            st.caption(f"{n} test slices. Filter settings were tuned on the validation patients, not the test patients.")
            show_image(d / "extras" / "baselines_visual.png", "Top: images. Bottom: absolute error against the routine-dose scan (0 to 60 HU); darker is better.")

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            show_image(d / "extras" / "1_residual.png", "What was removed (third panel) versus the true noise (fourth). Speckle, no anatomy.")
        with c2:
            show_image(d / "extras" / "2_spectrum.png", "Noise power spectrum: error is lower at every frequency, least so for slow blotchy noise.")
        st.caption("The notes below describe the reported abdomen run. If you retrain, re-check them against the figures above.")
        st.markdown("""
* The removed signal contains **no organ outlines**, so the model is taking out noise, not structure.
* The error drops about 4 to 5 times in power at mid and high frequencies, but only about 2 times at the lowest frequencies.
  Slow blotchy noise looks like real tissue, so it is the hardest to remove safely.
* The outputs look smoother than the routine-dose scan (a "waxy" look). That is a known side effect of L1-type losses.""")


def page_demo() -> None:
    st.title("Live demo")
    st.markdown("This page sends a slice to the **inference server**, which runs the trained model on the GPU and sends the enhanced slice back.")
    try:
        from client import LDCTClient, LDCTError
    except ImportError:
        st.error("client.py was not found next to app_site.py. Copy it from the serving package.")
        return
    with st.expander("Server connection", expanded=False):
        url = st.text_input("Address", os.environ.get("LDCT_SERVER", "http://127.0.0.1:8000"))
        key = st.text_input("API key (if the server needs one)", os.environ.get("LDCT_API_KEY", ""), type="password")
    client = LDCTClient(url, key or None, timeout=180)
    try:
        h = client.health()
        models = client.models()
    except LDCTError as exc:
        st.markdown(f'<div class="caution"><b>The server is not reachable.</b> {exc}<br>Start it in another terminal: '
                    '<code>uvicorn server:app --host 127.0.0.1 --port 8000</code>, then reload this page.</div>', unsafe_allow_html=True)
        st.graphviz_chart("""digraph G { rankdir=LR; bgcolor="transparent";
 node [shape=box, style="rounded,filled", fillcolor="#dcecf4", color="#1f7a8c", fontname="Helvetica", fontsize=11, fontcolor="#0b2540"];
 Client [label="This page\\n(no GPU needed)"]; Server [label="server.py\\nFastAPI"]; Reg [label="Model registry\\nbest.pt files"]; GPU [label="GPU inference"];
 Client -> Server [label=" HTTP + file "]; Server -> Reg -> GPU; GPU -> Server -> Client [label=" enhanced slice "]; }""")
        return
    st.success(f"Connected to the server on {h['device']}" + (f" ({h['gpu']})" if h.get("gpu") else "") + f" with {len(models)} model(s).")
    if not models:
        st.warning("The server has no trained models yet.")
        return

    c1, c2 = st.columns([1, 2])
    organs = sorted({m["organ"] for m in models})
    organ = c1.selectbox("Organ", organs, format_func=organ_title)
    mine = [m for m in models if m["organ"] == organ]
    keys = [m["arch"] for m in mine]
    labels = {m["arch"]: m["label"] + (" (default)" if m["default"] else "") for m in mine}
    default_arch = next((m["arch"] for m in mine if m["default"]), keys[0])
    compare = c2.toggle("Compare several models on the same slice", value=False)
    archs = (c2.multiselect("Models to run", keys, default=keys[: min(3, len(keys))], format_func=lambda a: labels[a]) if compare
             else [c2.selectbox("Model", keys, index=keys.index(default_arch), format_func=lambda a: labels[a])])
    up = st.file_uploader("Low-dose scan (.npy in HU, .dcm, .png)", type=["npy", "dcm", "png", "jpg", "jpeg"])
    ref_up = st.file_uploader("Optional: routine-dose reference (.npy) - adds PSNR / SSIM / RMSE", type=["npy"])
    if up is not None:
        data: object = up.getvalue()
        z = None
        if up.name.lower().endswith(".npy"):
            data = np.load(io.BytesIO(data), allow_pickle=False)
            st.caption(f"array shape {data.shape} - values are assumed to be Hounsfield units")
            if data.ndim == 3 and data.shape[0] > 1:
                z = st.slider("Slice", 0, data.shape[0] - 1, data.shape[0] // 2)
        else:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as fh:
                fh.write(data)
            data = fh.name
        reference = np.load(io.BytesIO(ref_up.getvalue()), allow_pickle=False) if ref_up is not None else None
        if st.button("Enhance on the server", type="primary", disabled=not archs):
            try:
                out, bar = {}, st.progress(0.0, text="Waiting for the server...")
                for i, a in enumerate(archs):
                    bar.progress(i / len(archs), text=f"Running {a} ({i + 1}/{len(archs)})...")
                    out[a] = client.enhance(data, organ, a, z=z, reference=reference)
                bar.empty()
                st.session_state["site_results"] = out
            except LDCTError as exc:
                st.error(str(exc))
        if isinstance(data, str):
            try:
                os.unlink(data)
            except OSError:
                pass

    results = st.session_state.get("site_results")
    if results:
        first = next(iter(results.values()))
        st.caption(f"slice {first.meta['slice']} of {first.meta['n_slices']} | {first.meta['shape'][0]}x{first.meta['shape'][1]} px")
        for r in results.values():
            if r.meta.get("warning"):
                st.warning(f"{r.meta['model']['label']}: {r.meta['warning']}")
        panels = [("Low dose (input)", first.input_png)]
        panels += [(f"{r.meta['model']['label']} ({r.meta['inference_ms']:.0f} ms)", r.png) for r in results.values()]
        if first.reference_png:
            panels.append(("Routine dose (reference)", first.reference_png))
        for i in range(0, len(panels), 4):
            for col, (title, png) in zip(st.columns(4), panels[i:i + 4]):
                col.image(png, caption=title)
        if first.metrics:
            k0 = first.metrics
            rows = [dict(Model="Low dose (input)", PSNR=k0["psnr_in"], SSIM=k0["ssim_in"], RMSE_HU=k0["rmse_in"])]
            rows += [dict(Model=r.meta["model"]["label"], PSNR=r.metrics["psnr_out"], SSIM=r.metrics["ssim_out"], RMSE_HU=r.metrics["rmse_out"])
                     for r in results.values()]
            table(pd.DataFrame(rows).round(3))
            st.caption("One slice only: this is a demonstration, not evidence. The Results page has the test-set evidence.")


def page_limits() -> None:
    st.title("Limitations and ethics")
    st.markdown("### What this project does not show")
    table(pd.DataFrame([
        ("Small patient count", "15 patients, all from one collection. Cross-validation tests all of them, but the group is small."),
        ("One vendor, one body region", "Siemens abdomen only. Other body regions and the GE scans were not used."),
        ("Simulated low dose", "The low-dose images are made by adding noise to raw data. Real low-dose scans may behave differently."),
        ("5 mm slices", "Thicker slices are easier to denoise than thin ones."),
        ("Metrics are not clinical quality", "PSNR, SSIM and RMSE do not measure whether a radiologist can still find a lesion."),
        ("No lesion study", "No test of whether small lesions stay visible. The model could smooth a faint one."),
        ("Waxy texture", "Outputs are smoother than the routine-dose reference."),
        ("Low-frequency noise remains", "Slow blotchy noise is only about halved."),
        ("Untuned design choices", "Loss weights were set by hand and no ablation was run, so we cannot say which design detail made the best model best."),
    ], columns=["Limitation", "What it means"]))
    st.markdown("### Next steps")
    st.markdown("""
1. Test generalisation on **other body regions** (for example chest) and on the GE scans.
2. Run **ablations**: 2D versus 2.5D, each loss term, the residual output.
3. A **lesion-detectability study** or a short reader study.
4. A **perceptual or adversarial loss** to address the waxy texture.
5. Several random **seeds**, so the small gaps between deep models carry error bars.""")
    st.markdown("### Ethics and use")
    st.markdown("""
* The data is **de-identified** and distributed under **CC BY 4.0**; it must be cited (see References).
* This is a **research prototype and is not a medical device**. Its output must never be used for diagnosis or treatment decisions.
* A denoiser can hide or invent small structures. Any clinical use would need validation against lesion detection with radiologists.""")


def page_reproduce() -> None:
    st.title("Reproduce and references")
    st.markdown("### Files")
    table(pd.DataFrame([
        ("dataset.py", "Organ settings, DICOM to HU conversion, patient splits, patch dataset"),
        ("model.py", "Architectures, loss, training, evaluation, `compare`"),
        ("extras.py", "Tuned classical baselines and explanation figures"),
        ("cv.py", "Patient-level cross-validation"),
        ("server.py / client.py", "Inference server and client library (client-server architecture)"),
        ("app_site.py", "This website"),
        ("app.py / app_client.py", "Local training app and thin client"),
        ("test_serving.py", "End-to-end test of the server and client"),
    ], columns=["File", "Role"]))
    st.markdown("### Commands, in order")
    st.code("""python dataset.py from-folder --organ liver --dicom-dir D:/ldct_dicom
python model.py compare --organ liver --epochs 30 --archs resunet unet dncnn redcnn cnn
python extras.py all --organ liver --arch resunet
python cv.py --organ liver --arch resunet --folds 5 --epochs 30
uvicorn server:app --host 127.0.0.1 --port 8000      # terminal 1
streamlit run app_site.py                            # terminal 2""", language="powershell")
    st.caption("Environment used: Windows, Python 3.12, PyTorch with CUDA, NVIDIA RTX 4060 Laptop GPU. Fixed random seed 42.")

    st.markdown("### Dataset citation")
    st.markdown("McCollough, C. H. et al. *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*. The Cancer Imaging Archive. "
                "https://doi.org/10.7937/9npb-2637. Licence: CC BY 4.0. Also cite: Clark, K. et al. The Cancer Imaging Archive (TCIA): maintaining and "
                "operating a public information repository. *Journal of Digital Imaging* 26(6), 1045-1057, 2013.")
    st.markdown("### References")
    for r in [
        "Zhang, K., Zuo, W., Chen, Y., Meng, D., Zhang, L. Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising. IEEE Trans. Image Processing 26(7), 2017.",
        "Chen, H. et al. Low-dose CT with a residual encoder-decoder convolutional neural network. IEEE Trans. Medical Imaging 36(12), 2017.",
        "Ronneberger, O., Fischer, P., Brox, T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.",
        "Dong, C., Loy, C. C., He, K., Tang, X. Image super-resolution using deep convolutional networks. IEEE Trans. PAMI 38(2), 2016.",
        "Buades, A., Coll, B., Morel, J.-M. A non-local algorithm for image denoising. CVPR 2005.",
        "Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Processing 13(4), 2004.",
        "Zamir, S. W. et al. Restormer: efficient transformer for high-resolution image restoration. CVPR 2022.",
    ]:
        st.markdown(f"* {r}")


# -------------------------------------------------------------------------------------------- navigation
PAGES = {"Overview": page_overview, "Background": page_background, "Data and scope": page_data, "Method": page_method,
         "Results": page_results, "Live demo": page_demo, "Limitations and ethics": page_limits, "Reproduce and references": page_reproduce}
with st.sidebar:
    st.markdown(f"## {PROJECT['name']}")
    st.caption(PROJECT["tagline"])
    choice = st.radio("Sections", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Research prototype. Not a medical device. Not for diagnosis.")
PAGES[choice]()
