"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone - Project 1
Run:  streamlit run app.py
"""
import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # app still works in demo mode
    genai = None

st.set_page_config(page_title="Construction Intelligence Platform", page_icon="🏗️", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.hero{background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);padding:22px 28px;border-radius:14px;color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:1.8rem}.hero p{margin:4px 0 0;opacity:.85}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,.06);color:#111}
.card .lbl{font-size:.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
.card .val{font-size:1.6rem;font-weight:700;color:#111}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.8rem;font-weight:600;color:#fff}
.Low,.COMPLIANT,.Routine{background:#16a34a}.Medium,.PARTIAL,.Monitor{background:#d97706}
.High,.NON-COMPLIANT,.Urgent{background:#ea580c}.Critical,.Immediate{background:#dc2626}
.alert{border-left:5px solid #dc2626;background:#fef2f2;padding:10px 14px;border-radius:6px;margin:6px 0;color:#7f1d1d}
.alert.ok{border-color:#16a34a;background:#f0fdf4;color:#14532d}
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def pill(text: str) -> str:
    cls = str(text).replace(" ", "-")
    return f'<span class="pill {cls}">{text}</span>'


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
def get_model():
    key = st.session_state.get("api_key", "").strip()
    if not key or genai is None:
        return None
    try:
        genai.configure(api_key=key)
        return genai.GenerativeModel(st.session_state.get("model_name", "gemini-2.0-flash"))
    except Exception as e:
        st.sidebar.error(f"Gemini init failed: {e}")
        return None


def parse_json(text: str) -> dict:
    """Extract a JSON object from an LLM response (handles ``` fences)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def analyze_image(prompt: str, image: Image.Image, demo: dict) -> tuple[dict, bool]:
    """Returns (result, is_demo). Falls back to demo data on any failure."""
    model = get_model()
    if model is None:
        return demo, True
    try:
        resp = model.generate_content([prompt, image])
        return parse_json(resp.text), False
    except Exception as e:
        st.warning(f"Live analysis failed ({e}). Showing demonstration output.")
        return demo, True


# ----------------------------------------------------------------------------
# Prompts (JSON data contracts) and demo payloads
# ----------------------------------------------------------------------------
PPE_PROMPT = """You are a certified construction safety auditor (OSHA 1926 and IS 456 / NBC India).
Analyze the site photo for worker PPE compliance and site hazards.
Respond with ONLY valid JSON matching this contract:
{"safety_score": int 0-100,
 "overall_status": "COMPLIANT"|"PARTIAL"|"NON-COMPLIANT",
 "workers_detected": int,
 "ppe_detection": {"helmet": bool, "high_vis_vest": bool, "steel_toe_boots": bool, "safety_harness": bool},
 "hazards": [{"hazard": str, "severity": "Low"|"Medium"|"High"|"Critical", "remedy": str}],
 "summary": str}
If PPE item is not visible or not required, set false and mention it in the summary."""

CRACK_PROMPT = """You are a structural engineer diagnosing concrete defects (IS 456, IS 15988).
Analyze the image. Respond with ONLY valid JSON:
{"defect_type": "Hairline"|"Settlement"|"Structural Shear"|"Thermal Expansion"|"Spalling",
 "severity_score": int 0-100,
 "estimated_width_mm": float,
 "urgency": "Routine"|"Monitor"|"Urgent"|"Immediate",
 "root_causes": [str],
 "recommended_repairs": [str],
 "summary": str}"""

DEMO_PPE = {
    "safety_score": 62, "overall_status": "PARTIAL", "workers_detected": 3,
    "ppe_detection": {"helmet": True, "high_vis_vest": True, "steel_toe_boots": False, "safety_harness": False},
    "hazards": [
        {"hazard": "Worker at height without fall-arrest harness", "severity": "Critical",
         "remedy": "Stop work at height; provide full-body harness anchored to a certified lifeline."},
        {"hazard": "Open footwear on rebar-strewn surface", "severity": "High",
         "remedy": "Mandate ISI-marked steel-toe boots; enforce at gate checks."},
        {"hazard": "Unorganised material stacking", "severity": "Medium",
         "remedy": "Designate laydown zones and keep walkways 1 m clear."},
    ],
    "summary": "[DEMO] Helmets and vests observed, but footwear and fall protection are deficient.",
}
DEMO_CRACK = {
    "defect_type": "Settlement", "severity_score": 68, "estimated_width_mm": 3.2, "urgency": "Urgent",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction", "Water ingress at footing"],
    "recommended_repairs": ["Install crack gauges and monitor 4 weeks", "Epoxy injection (IS 15988)",
                            "Structural re-grouting / underpinning if movement continues"],
    "summary": "[DEMO] Diagonal stepped crack consistent with differential settlement.",
}

# ----------------------------------------------------------------------------
# Valuation model (illustrative INR rates - editable)
# ----------------------------------------------------------------------------
COST_PER_SQFT = {"RCC Framed": 2200, "Load-Bearing Masonry": 1700, "Steel / PEB": 2600, "Luxury Villa": 3800}
MARKET_PER_SQFT = {"Tier 1": 9500, "Tier 2": 5200, "Tier 3": 3200}
LOC_COST_FACTOR = {"Tier 1": 1.15, "Tier 2": 1.0, "Tier 3": 0.9}
STEEL_KG_SQFT = {"RCC Framed": 4.0, "Load-Bearing Masonry": 2.2, "Steel / PEB": 7.5, "Luxury Villa": 4.8}
CEMENT_BAGS_SQFT = {"RCC Framed": 0.45, "Load-Bearing Masonry": 0.40, "Steel / PEB": 0.25, "Luxury Villa": 0.55}
MILESTONES = [(10, "Site prep & excavation"), (25, "Foundation"), (45, "Superstructure"),
              (65, "Masonry & roofing"), (85, "MEP & plastering"), (100, "Finishing & handover")]


def inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    return f"₹{v / 1e5:,.2f} Lakhs"


# ----------------------------------------------------------------------------
# Weather protocols
# ----------------------------------------------------------------------------
WEATHER = {
    "Normal": (True, ["Standard toolbox talk at 8:00 AM.", "Routine PPE and scaffold inspection before shift.",
                      "Cure concrete by ponding / wet hessian for minimum 7 days (IS 456)."]),
    "High Wind": (False, ["SUSPEND tower crane and hoist operations above 36 km/h wind speed.",
                          "Secure loose sheets, formwork and scaffolding netting.",
                          "Halt work at height; evacuate open edges and roof slabs."]),
    "Heavy Rain": (False, ["Isolate temporary electrical panels; de-energise wet areas (shock hazard).",
                           "Cover fresh concrete with tarpaulin; avoid pours during downpour.",
                           "Pump out excavations and inspect trench shoring for collapse risk."]),
    "Extreme Heat": (False, ["Shift heavy work to early morning; mandatory hydration breaks every 45 min.",
                             "Start concrete curing early: fogging, wet jute, curing compound to prevent plastic cracking.",
                             "Monitor workers for heat stress (ORS and shaded rest zones)."]),
}


def demo_chat_reply(q: str) -> str:
    q = q.lower()
    if "cure" in q or "curing" in q:
        return "Keep concrete continuously moist for at least 7 days (OPC) or 10 days (PPC/blended) as per IS 456 clause 13.5."
    if "mix" in q or "m25" in q:
        return "M25 nominal ratio is roughly 1 : 1 : 2 (cement : sand : aggregate), but design mix with a lab test is preferred."
    if "crack" in q:
        return "Measure width and monitor with tell-tales. Hairline (<0.3 mm) is usually cosmetic; wider or stepped cracks need an engineer's review."
    if "helmet" in q or "ppe" in q or "safety" in q:
        return "Minimum PPE: IS 2925 helmet, hi-vis vest, IS 15298 safety boots, gloves, and a harness above 1.8 m (OSHA 1926.501)."
    return "[Demo mode] Add a Gemini API key in the sidebar for full AI answers. Try asking about curing, mixes, cracks or PPE."


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔐 Configuration")
    st.session_state["api_key"] = st.text_input("Google Gemini API key", type="password",
                                                help="Leave empty to use demonstration mode.")
    st.session_state["model_name"] = st.text_input("Model", value="gemini-2.0-flash")
    live = bool(st.session_state["api_key"].strip()) and genai is not None
    st.success("Live AI mode" if live else "Demonstration mode (no key)")
    st.divider()
    st.subheader("🧭 Architecture")
    st.markdown(
        "- **UI:** Streamlit\n- **Vision/LLM:** Gemini (multimodal)\n- **Contracts:** JSON schemas\n"
        "- **Valuation:** rule-based INR engine\n- **Render:** Pollinations AI\n- **State:** session only, no DB"
    )
    st.subheader("📌 Features")
    st.markdown("1. PPE safety audit\n2. Crack diagnostics\n3. Cost & valuation\n4. Weather advisory + chatbot")

st.markdown(
    '<div class="hero"><h1>🏗️ Multimodal Construction Intelligence Platform</h1>'
    "<p>AI-powered safety audits, structural diagnostics, INR valuation and weather-aware site protocols.</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["🦺 Safety & PPE Audit", "🧱 Crack Diagnostics", "💰 Planning & Valuation",
                                  "🌦️ Site Assistant"])

# ----------------------------------------------------------------------------
# Tab 1: PPE
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Site Safety & PPE Compliance Audit")
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="ppe_up")
    if up:
        try:
            img = Image.open(up).convert("RGB")
        except Exception:
            st.error("Could not read the image file.")
            img = None
        if img:
            c1, c2 = st.columns([1, 1.3])
            c1.image(img, caption="Uploaded site photo", use_container_width=True)
            if c2.button("Run safety audit", type="primary", key="ppe_btn"):
                with st.spinner("Analyzing PPE compliance..."):
                    st.session_state["ppe"] = analyze_image(PPE_PROMPT, img, DEMO_PPE)
            if "ppe" in st.session_state:
                res, is_demo = st.session_state["ppe"]
                with c2:
                    if is_demo:
                        st.info("Demonstration output (no live API call).")
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(card("Safety score", f'{res.get("safety_score", 0)}/100'), unsafe_allow_html=True)
                    m2.markdown(card("Workers", str(res.get("workers_detected", "-"))), unsafe_allow_html=True)
                    m3.markdown(card("Status", pill(res.get("overall_status", "PARTIAL"))), unsafe_allow_html=True)
                    st.progress(min(max(int(res.get("safety_score", 0)), 0), 100) / 100)
                    st.markdown("**PPE checklist**")
                    for k, v in res.get("ppe_detection", {}).items():
                        st.write(("✅" if v else "❌") + " " + k.replace("_", " ").title())
                st.markdown("### Identified hazards")
                for h in res.get("hazards", []):
                    st.markdown(f'{pill(h.get("severity", "Low"))} **{h.get("hazard", "")}**  \n'
                                f'↳ _Remedy:_ {h.get("remedy", "")}', unsafe_allow_html=True)
                st.caption(res.get("summary", ""))
                st.download_button("⬇️ Download JSON report", json.dumps(res, indent=2),
                                   file_name=f"ppe_report_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")
    else:
        st.info("Upload a JPG/PNG to begin. Without an API key, sample output is shown.")

# ----------------------------------------------------------------------------
# Tab 2: Cracks
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    up2 = st.file_uploader("Upload concrete / wall surface photo", type=["jpg", "jpeg", "png"], key="crack_up")
    if up2:
        try:
            img2 = Image.open(up2).convert("RGB")
        except Exception:
            st.error("Could not read the image file.")
            img2 = None
        if img2:
            c1, c2 = st.columns([1, 1.3])
            c1.image(img2, caption="Surface under inspection", use_container_width=True)
            if c2.button("Diagnose defect", type="primary", key="crack_btn"):
                with st.spinner("Running structural diagnostics..."):
                    st.session_state["crack"] = analyze_image(CRACK_PROMPT, img2, DEMO_CRACK)
            if "crack" in st.session_state:
                r, is_demo = st.session_state["crack"]
                with c2:
                    if is_demo:
                        st.info("Demonstration output (no live API call).")
                    a, b, c = st.columns(3)
                    a.markdown(card("Defect", r.get("defect_type", "-")), unsafe_allow_html=True)
                    b.markdown(card("Width", f'{r.get("estimated_width_mm", 0)} mm'), unsafe_allow_html=True)
                    c.markdown(card("Urgency", pill(r.get("urgency", "Monitor"))), unsafe_allow_html=True)
                    st.markdown(f"**Severity score:** {r.get('severity_score', 0)}/100")
                    st.progress(min(max(int(r.get("severity_score", 0)), 0), 100) / 100)
                    st.markdown("**Root causes**")
                    for x in r.get("root_causes", []):
                        st.write("• " + x)
                    st.markdown("**Recommended engineering repairs**")
                    for x in r.get("recommended_repairs", []):
                        st.write("🔧 " + x)
                    st.caption(r.get("summary", ""))
                st.download_button("⬇️ Download JSON report", json.dumps(r, indent=2),
                                   file_name=f"crack_report_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")
    else:
        st.info("Upload a photo of a crack or damaged concrete surface.")

# ----------------------------------------------------------------------------
# Tab 3: Valuation
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Project Planning & Real Estate Valuation (INR)")
    c1, c2 = st.columns(2)
    stype = c1.selectbox("Structure type", list(COST_PER_SQFT))
    tier = c2.selectbox("Location tier", list(MARKET_PER_SQFT), format_func=lambda t: f"{t} city")
    sqft = c1.number_input("Built-up area (sq ft)", 200, 500000, 2000, step=100)
    pct = c2.slider("Completion (%)", 0, 100, 40)

    budget = sqft * COST_PER_SQFT[stype] * LOC_COST_FACTOR[tier]
    capex = budget * pct / 100
    # Asset value grows with completion; fully built asset is worth full market rate
    market_full = sqft * MARKET_PER_SQFT[tier]
    asset_val = market_full * (0.35 + 0.65 * pct / 100) if pct > 0 else market_full * 0.35  # land share + built share
    steel_t = sqft * STEEL_KG_SQFT[stype] / 1000
    cement = sqft * CEMENT_BAGS_SQFT[stype]

    m = st.columns(4)
    m[0].markdown(card("Total budget", inr(budget)), unsafe_allow_html=True)
    m[1].markdown(card("CapEx incurred", inr(capex)), unsafe_allow_html=True)
    m[2].markdown(card("Current asset value", inr(asset_val)), unsafe_allow_html=True)
    m[3].markdown(card("Remaining spend", inr(budget - capex)), unsafe_allow_html=True)
    m2 = st.columns(2)
    m2[0].markdown(card("Steel requirement", f"{steel_t:,.1f} tonnes"), unsafe_allow_html=True)
    m2[1].markdown(card("Cement requirement", f"{cement:,.0f} bags (50 kg)"), unsafe_allow_html=True)

    st.markdown("#### Milestone progress")
    st.progress(pct / 100)
    for thr, name in MILESTONES:
        st.write(("✅ " if pct >= thr else "⬜ ") + f"{name} ({thr}%)")
    st.caption("Rates are illustrative planning estimates; validate with local schedule of rates (CPWD/DSR).")

    st.markdown("#### 🎨 Conceptual 3D architectural rendering")
    style = st.text_input("Style", "modern minimalist with glass facade and landscaped garden")
    storeys = st.selectbox("Storeys", ["G+0", "G+1", "G+2", "G+4"], index=1)
    p = (f"photorealistic 3D architectural render of a {storeys} {stype} building, {style}, "
         f"Indian {tier.lower()} city, golden hour, high detail")
    st.code(p, language=None)
    if st.button("Generate rendering"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}?width=1024&height=640&nologo=true"
        try:
            st.image(url, caption="AI concept render (Pollinations)", use_container_width=True)
        except Exception as e:
            st.error(f"Render service unavailable: {e}")

# ----------------------------------------------------------------------------
# Tab 4: Weather + chatbot
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Weather-Aware Daily Safety Advisory")
    w = st.radio("Today's condition", list(WEATHER), horizontal=True)
    ok, items = WEATHER[w]
    for it in items:
        st.markdown(f'<div class="alert {"ok" if ok else ""}">{it}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("💬 Construction Engineering Assistant")
    if "chat" not in st.session_state:
        st.session_state["chat"] = [{"role": "assistant",
                                     "content": "Hi! Ask me about curing, mixes, IS codes, safety or scheduling."}]
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if q := st.chat_input("Ask a site engineering question..."):
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        model = get_model()
        try:
            if model is None:
                ans = demo_chat_reply(q)
            else:
                history = "\n".join(f'{m["role"]}: {m["content"]}' for m in st.session_state["chat"][-8:])
                sys = (f"You are a senior civil/site engineer in India. Current weather: {w}. "
                       "Be concise, cite IS codes/OSHA where relevant, prioritise safety.\n")
                ans = model.generate_content(sys + history).text
        except Exception as e:
            ans = f"Sorry, the AI service returned an error: {e}"
        st.session_state["chat"].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)
    if st.button("Clear chat"):
        del st.session_state["chat"]
        st.rerun()
