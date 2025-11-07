import os, math
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ── MUST be the first Streamlit call ───────────────────────────────────────────
st.set_page_config(page_title="RC Beam Rebar Assistant",
                   page_icon="🧱",
                   layout="centered")

# Optional: load OPENAI_API_KEY from .env if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============== CONFIG ==================
FEATURES = ["Width_mm", "Depth_mm", "Moment_kNm", "Fck"]
MODEL_PATH = "model/beam_bottom_bar_model.pkl"
TRAIN_DOMAIN = {
    "width":  (125, 300),
    "depth":  (450, 600),
    "moment": (2, 318),
    "fck":    (25, 25)   # constant in current dataset
}

# ============== LOAD MODEL ===============
@st.cache_resource
def load_model():
    """Load the trained sklearn Pipeline (.pkl). Cached across reruns."""
    pipe = joblib.load(MODEL_PATH)
    return pipe

model = load_model()

# ============== HELPERS ==================
def predict_as(width_mm, depth_mm, moment_kNm, fck=25.0, pipe=model):
    """Predict required As (mm²) using the trained pipeline."""
    X = pd.DataFrame([[width_mm, depth_mm, moment_kNm, fck]], columns=FEATURES)
    as_req = float(pipe.predict(X)[0])
    return as_req

def bar_options(required_area_mm2, diameters=(10,12,16,20,25,32), max_bars=8,
                beam_width_mm=None, clear_cover_mm=25, min_clear_spacing_mm=None):
    """
    Enumerate single-diameter options that meet/exceed As.
    Sorted by: lowest over-provision %, then fewest bars, then smaller dia.
    If beam_width_mm is provided, also flag if the set fits given cover/spacing.
    """
    out = []
    for d in diameters:
        area_one = math.pi * (d**2) / 4.0
        for n in range(2, max_bars + 1):
            total = area_one * n
            if total >= required_area_mm2:
                over = 100.0 * (total - required_area_mm2) / required_area_mm2
                fits = True
                if beam_width_mm is not None:
                    spacing = max(d, (min_clear_spacing_mm or max(25, d)))
                    usable = beam_width_mm - 2 * clear_cover_mm
                    required_width = n * d + (n - 1) * spacing
                    fits = required_width <= usable
                out.append({
                    "Notation": f"{n}T{d}",
                    "Bars": n,
                    "Dia_mm": d,
                    "Provided_Area_mm2": round(total, 1),
                    "Overprovision_%": round(over, 1),
                    "Fits_Width": fits
                })
                break
    out = sorted(out, key=lambda r: (r["Overprovision_%"], r["Bars"], r["Dia_mm"]))
    return pd.DataFrame(out)

def in_training_scope(b, d, m, fck):
    """Check if inputs fall within the dataset’s training domain."""
    return (TRAIN_DOMAIN["width"][0]  <= b <= TRAIN_DOMAIN["width"][1]  and
            TRAIN_DOMAIN["depth"][0]  <= d <= TRAIN_DOMAIN["depth"][1]  and
            TRAIN_DOMAIN["moment"][0] <= m <= TRAIN_DOMAIN["moment"][1] and
            abs(fck - TRAIN_DOMAIN["fck"][0]) < 1e-9)

# ============== UI =======================
st.title("🧱 RC Beam Rebar Assistant (ML + GPT)")

st.markdown("""
This tool estimates **required bottom reinforcement area (As, mm²)** using a trained ML model  
and proposes **bar sets** (e.g., `3T12`) that meet or exceed As with minimal over-provision.

> **Scope (training domain):** Width **125–300 mm**, Depth **450–600 mm**, Sagging Moment **2–318 kNm**, **Fck = 25 MPa**.
""")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    width_mm  = col1.number_input("Beam width b (mm)",  min_value=50,  max_value=1500, value=150, step=5)
    depth_mm  = col2.number_input("Beam depth h (mm)",  min_value=200, max_value=2000, value=500, step=10)
    moment_k  = st.number_input("Sagging Moment Mu (kNm)", min_value=0.0, max_value=2000.0, value=120.0, step=1.0)
    fck       = st.number_input("Concrete grade fck (MPa)", min_value=10.0, max_value=80.0, value=25.0, step=5.0)

    st.markdown("**Optional:** provide beam width for spacing/fit check")
    beam_width_for_fit = st.number_input("Beam width for fit check (mm)", min_value=0, max_value=2000, value=int(width_mm), step=5)
    clear_cover_mm     = st.number_input("Clear cover (mm)", min_value=15, max_value=75, value=25, step=5)
    min_spacing_mm     = st.number_input("Min clear spacing (mm) (blank = max(25,dia))", min_value=0, max_value=100, value=0, step=5)

    run_btn = st.form_submit_button("Predict & Propose Bars")

if run_btn:
    # 1) Domain guard
    if not in_training_scope(width_mm, depth_mm, moment_k, fck):
        st.warning(
            "⚠️ Input is outside the training scope: Width 125–300, Depth 450–600, "
            "Moment 2–318, fck=25 MPa. Prediction may be unreliable; consider retraining with broader data."
        )

    # 2) Predict As
    as_req = predict_as(width_mm, depth_mm, moment_k, fck)
    st.subheader(f"Predicted Required As: **{as_req:.1f} mm²**")

    # 3) Bar options table
    spacing = (min_spacing_mm if min_spacing_mm > 0 else None)
    opts_df = bar_options(as_req,
                          beam_width_mm=beam_width_for_fit,
                          clear_cover_mm=clear_cover_mm,
                          min_clear_spacing_mm=spacing)
    st.markdown("### Bar options (ranked by lowest overprovision)")
    st.dataframe(opts_df)

    # 4) GPT explanation (optional)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            sys = (
                "You are an RC beam reinforcement copilot. "
                "Explain the ML-predicted As (mm²) and bar options in plain language. "
                "Be concise but technical. "
                "Mention training scope limits and include a safety note to verify with the relevant design code "
                "(e.g., EC2/BS 8110) for minimum steel, spacing, anchorage and ductility."
            )
            user = (
                f"Inputs: width={width_mm} mm, depth={depth_mm} mm, Mu={moment_k} kNm, fck={fck} MPa. "
                f"Predicted As = {as_req:.1f} mm². "
                f"Top 3 options: {opts_df.head(3).to_dict(orient='records')}. "
                f"Beam width for fit check={beam_width_for_fit} mm, clear cover={clear_cover_mm} mm, "
                f"min clear spacing used={spacing if spacing else 'max(25, dia)'}."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=0.2
            )
            text = resp.choices[0].message.content
            st.markdown("### GPT Explanation")
            st.write(text)
        except Exception as e:
            st.info("GPT explanation unavailable. Check your `OPENAI_API_KEY` or try again.")
            st.caption(f"Details: {e}")
    else:
        st.info("Set your OPENAI_API_KEY to get GPT explanations.")
        st.caption("Tip: create a .env with OPENAI_API_KEY=sk-... or set it in your Streamlit Cloud secrets.")

st.divider()
st.caption(
    "Disclaimer: Educational support only. Verify final designs against the relevant code (e.g., EC2/BS 8110) "
    "for minimum steel, spacing, anchorage and ductility. "
    "Model scope is limited to training ranges (Width 125–300 mm, Depth 450–600 mm, Moment 2–318 kNm, Fck=25 MPa)."
)
