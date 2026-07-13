# app.py — TAP CSR Research Agent v5 — Two-Spoke Streamlit UI (2026-07)
"""
TWO MODES:
  🔍 Prospect Screening  — fast (< 60s), binary qualify/don't qualify
  🔬 Deep Research       — full investigation, 6 sources, DOCX + HTML report

Run with:
    streamlit run app.py
"""

import json
import sys
import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads ANTHROPIC_API_KEY etc. from .env into os.environ
except ImportError:
    pass  # python-dotenv not installed; env vars can still be set manually

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper       import fetch_screen_sources, fetch_deep_sources
from parser        import parse_all
from scorer        import score as compute_score
from reporter      import generate_html_report
from docx_reporter import generate_docx_report


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TAP CSR Research Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
# Uses Streamlit's own theme CSS variables so every custom component adapts
# automatically when the user switches Light ⇄ Dark in Settings.
st.markdown("""
<style>
  :root{
    --tap-purple:#7C3AED;
    --tap-purple-soft:color-mix(in srgb,#7C3AED 12%,transparent);
    --tap-purple-border:color-mix(in srgb,#7C3AED 35%,transparent);
    --tap-green:#16A34A; --tap-amber:#D97706; --tap-red:#DC2626;
    --tap-ink:var(--text-color,#1A1A2E);
    --tap-card:var(--secondary-background-color,#F4EFE8);
    --tap-muted:color-mix(in srgb,var(--text-color,#1A1A2E) 55%,transparent);
    --tap-line:color-mix(in srgb,var(--text-color,#1A1A2E) 12%,transparent);
  }

  /* Hero */
  .tap-hero h1{font-size:2.1rem;font-weight:800;letter-spacing:-0.5px;
               line-height:1.15;margin:0;color:var(--tap-ink)}
  .tap-hero .accent{background:linear-gradient(90deg,#7C3AED,#A78BFA);
               -webkit-background-clip:text;background-clip:text;color:transparent}
  .tap-hero .sub{color:var(--tap-muted);font-size:0.95rem;margin-top:6px}
  .tap-chip{display:inline-flex;align-items:center;gap:6px;
            border:1px solid var(--tap-purple-border);
            background:var(--tap-purple-soft);color:var(--tap-purple);
            border-radius:999px;padding:3px 12px;font-size:0.75rem;
            font-weight:600;margin-bottom:10px}
  .tap-check{color:var(--tap-muted);font-size:0.8rem;margin-right:16px}
  .tap-check b{color:var(--tap-green)}

  /* Cards */
  .tap-card{background:var(--tap-card);border:1px solid var(--tap-line);
            border-radius:16px;padding:20px 22px}
  .score-card{text-align:center;border-radius:16px;padding:26px 16px;
              border:1px solid var(--tap-line)}
  .score-num{font-size:4rem;font-weight:800;line-height:1}
  .insight-box{background:var(--tap-purple-soft);
               border-left:4px solid var(--tap-purple);
               padding:16px 18px;border-radius:0 12px 12px 0;
               font-size:0.95rem;line-height:1.7;color:var(--tap-ink)}
  .cluster-card{background:color-mix(in srgb,var(--tap-green) 8%,transparent);
                border:1px solid color-mix(in srgb,var(--tap-green) 30%,transparent);
                border-radius:12px;padding:12px 14px;margin-bottom:8px}
  .ev-box{background:color-mix(in srgb,#F59E0B 10%,transparent);
          border-left:3px solid #F59E0B;
          padding:8px 12px;font-size:0.78rem;
          color:color-mix(in srgb,var(--text-color,#1A1A2E) 75%,#B45309);
          font-style:italic;border-radius:0 8px 8px 0;margin-top:6px}

  /* Screening verdict cards */
  .screen-qualify,.screen-review,.screen-skip{border-radius:16px;padding:22px;
      text-align:center;border:2px solid}
  .screen-qualify{background:color-mix(in srgb,var(--tap-green) 10%,transparent);
                  border-color:var(--tap-green)}
  .screen-review{background:color-mix(in srgb,var(--tap-amber) 10%,transparent);
                 border-color:var(--tap-amber)}
  .screen-skip{background:color-mix(in srgb,var(--tap-red) 10%,transparent);
               border-color:var(--tap-red)}
  .screen-pass{font-size:1.5rem;font-weight:800;color:var(--tap-green)}
  .screen-watchlist{font-size:1.5rem;font-weight:800;color:var(--tap-amber)}
  .screen-no{font-size:1.5rem;font-weight:800;color:var(--tap-red)}

  /* Pills & badges */
  .tap-pill{display:inline-block;background:var(--tap-purple-soft);
            color:var(--tap-purple);border-radius:999px;padding:2px 12px;
            font-size:0.78rem;font-weight:600;margin:2px 3px}
  .tap-pill-green{display:inline-block;border-radius:999px;padding:2px 12px;
            font-size:0.78rem;font-weight:600;margin:2px 3px;
            background:color-mix(in srgb,var(--tap-green) 12%,transparent);
            color:var(--tap-green)}
  .partner-row{padding:8px 0;border-bottom:1px solid var(--tap-line);
               font-size:0.9rem;color:var(--tap-ink)}

  /* Buttons */
  .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{
      border-radius:999px !important;font-weight:600 !important}
  .stFormSubmitButton>button{background:var(--tap-purple) !important;
      color:#fff !important;border:none !important}

  /* Metric-style steps */
  .step-card{background:var(--tap-card);border:1px solid var(--tap-line);
             border-radius:14px;padding:14px 16px;height:100%;min-height:150px}
  .step-card .n{color:var(--tap-purple);font-weight:800;font-size:0.8rem}
  .step-card .t{font-weight:700;font-size:0.9rem;color:var(--tap-ink);margin:2px 0}
  .step-card .d{font-size:0.78rem;color:var(--tap-muted);line-height:1.5}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_color(s):
    """5-tier strict gradient: 90+ priority, 80+ strong, 65+ promising,
    45+ watchlist, below = low fit."""
    return ("#7C3AED" if s >= 90 else "#16A34A" if s >= 80 else
            "#0EA5E9" if s >= 65 else "#D97706" if s >= 45 else "#DC2626")

def score_label(s):
    return ("PRIORITY MATCH" if s >= 90 else "STRONG MATCH" if s >= 80 else
            "PROMISING" if s >= 65 else "WATCHLIST" if s >= 45 else "LOW FIT")

def delivery_badge(delivery):
    """FUNDER / IMPLEMENTER / HYBRID badge — key pre-sales filter."""
    model = (delivery or {}).get("model", "UNCLEAR")
    m = {"FUNDER":      ("#16A34A", "🤝 FUNDER — grants to NGO partners"),
         "HYBRID":      ("#7C3AED", "🔀 HYBRID — funds partners + runs own programmes"),
         "IMPLEMENTER": ("#D97706", "🏗 IMPLEMENTER — runs CSR in-house"),
         "UNCLEAR":     ("#6B7280", "❔ Delivery model unclear — probe in outreach")}
    c, l = m.get(model, m["UNCLEAR"])
    return (f'<span style="background:{c};color:#fff;padding:4px 14px;'
            f'border-radius:999px;font-size:0.82em;font-weight:600;'
            f'margin-left:6px">{l}</span>')

def state_badge(state):
    m = {"FOUND":("✅","#16A34A","Evidence Found"),
         "NOT_FOUND_IN_SOURCE":("⚠️","#D97706","Partial Research"),
         "CONFIRMED_ABSENT":("❌","#DC2626","No CSR Evidence")}
    i, c, l = m.get(state, ("ℹ️","#6B7280",state))
    return (f'<span style="background:{c};color:#fff;padding:4px 14px;'
            f'border-radius:999px;font-size:0.82em;font-weight:600">{i} {l}</span>')

def dim_label(d):
    return {"focus_alignment":"🎯 Focus Alignment","adjacency_boost":"🔗 Adjacency Boost",
            "geography_fit":"📍 Geography","csr_maturity":"📋 CSR Maturity",
            "budget_size":"💰 Budget","source_quality":"🔍 Source Quality"}.get(d, d)


# ─────────────────────────────────────────────────────────────────────────────
# Theme switcher — Light (default) ⇄ Dark, sun/moon toggle in the top bar
# ─────────────────────────────────────────────────────────────────────────────

_THEMES = {
    "Light": {
        "base": "light", "primaryColor": "#7C3AED",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F6F5FA", "textColor": "#1A1A2E",
    },
    "Dark": {
        "base": "dark", "primaryColor": "#A78BFA",
        "backgroundColor": "#12121A",
        "secondaryBackgroundColor": "#1C1C28", "textColor": "#ECECF4",
    },
}

def _apply_theme(choice: str):
    """Set Streamlit theme options at runtime."""
    from streamlit import config as _st_config
    opts = _THEMES[choice]
    for k, v in opts.items():
        _st_config.set_option(f"theme.{k}", v)

# Light is the default (also shipped in .streamlit/config.toml)
st.session_state.setdefault("_applied_theme", "Light")

# Dynamic palette — set from Python so every custom component (hero, cards,
# pills) flips correctly. Dark mode: dark fonts become white.
_dark_on = st.session_state["_applied_theme"] == "Dark"
_ink     = "#ECECF4" if _dark_on else "#1A1A2E"
_card    = "#1C1C28" if _dark_on else "#F7F6FB"
_muted   = "rgba(236,236,244,.62)" if _dark_on else "rgba(26,26,46,.55)"
_line    = "rgba(236,236,244,.14)" if _dark_on else "rgba(26,26,46,.12)"
_purple  = "#A78BFA" if _dark_on else "#7C3AED"
st.markdown(f"""
<style>
  :root{{
    --tap-ink:{_ink}; --tap-card:{_card}; --tap-muted:{_muted};
    --tap-line:{_line}; --tap-purple:{_purple};
    --tap-purple-soft:color-mix(in srgb,{_purple} {"18" if _dark_on else "10"}%,transparent);
    --tap-purple-border:color-mix(in srgb,{_purple} 35%,transparent);
  }}
  .ev-box{{color:{"#FBBF24" if _dark_on else "#92400E"} !important}}
  /* theme switch pill — icon inside the knob, pinned beside Deploy */
  .st-key-theme_toggle{{position:fixed;top:0.75rem;right:7.5rem;
      width:64px;z-index:1000000;background:transparent}}
  .st-key-theme_toggle [data-testid="stElementContainer"]{{margin:0}}
  .st-key-theme_toggle button{{
      width:58px;height:30px;min-height:30px;border-radius:999px;
      padding:0 3px;display:flex;align-items:center;
      justify-content:{"flex-end" if _dark_on else "flex-start"};
      background:{"#3A3A48" if _dark_on else "#E9E5F6"} !important;
      border:1px solid {"rgba(236,236,244,.18)" if _dark_on else "rgba(26,26,46,.14)"} !important;
      transition:all .2s ease}}
  .st-key-theme_toggle button:hover{{border-color:{_purple} !important}}
  .st-key-theme_toggle button p{{margin:0;line-height:1;display:flex}}
  .st-key-theme_toggle button [data-testid="stIconMaterial"]{{
      width:24px;height:24px;border-radius:50%;font-size:17px;
      display:flex;align-items:center;justify-content:center;
      background:{"#15151F" if _dark_on else "#FFFFFF"};
      color:{"#ECECF4" if _dark_on else "#D97706"}}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="tap-chip">✨ AI-Powered CSR Intelligence</div>'
        '<h2 style="margin:0 0 2px">🎯 TAP CSR Agent</h2>'
        '<div style="font-size:0.85rem;opacity:0.7">The Apprentice Project</div>',
        unsafe_allow_html=True)
    st.divider()
    st.markdown("**Scoring (100 pts)**")
    st.caption("🎯 Focus Alignment — 40 pts")
    st.caption("🔗 Adjacency Boost — 20 pts")
    st.caption("📍 Geography — 10 pts")
    st.caption("📋 CSR Maturity — 10 pts")
    st.caption("💰 Budget — 10 pts")
    st.caption("🔍 Source Quality — 10 pts")
    st.divider()
    st.markdown("**Verdict bands (strict)**")
    st.caption("🟣 90+ Priority — engage now")
    st.caption("🟢 80–89 Strong — pursue actively")
    st.caption("🔵 65–79 Promising — nurture first")
    st.caption("🟠 45–64 Watchlist — revisit quarterly")
    st.caption("🔴 <45 Low fit — deprioritise")
    st.divider()
    st.caption("v6 · Aligned to TAP Buddy (21st-century skills) · "
               "Zero hallucination · Every claim cited & re-verified")


# ─────────────────────────────────────────────────────────────────────────────
# Main panel — theme toggle pinned in the header (beside Deploy), then hero
# ─────────────────────────────────────────────────────────────────────────────

with st.container(key="theme_toggle"):
    _is_dark = st.session_state["_applied_theme"] == "Dark"
    if st.button(
        ":material/dark_mode:" if _is_dark else ":material/light_mode:",
        key="theme_btn",
        help="Switch between light and dark mode",
    ):
        _want = "Light" if _is_dark else "Dark"
        _apply_theme(_want)
        st.session_state["_applied_theme"] = _want
        st.rerun()

st.markdown("""
<div class="tap-hero">
  <div class="tap-chip">✨ AI-Powered CSR Intelligence</div>
  <h1>Find the right CSR partners.<br><span class="accent">Create real impact.</span></h1>
  <div style="margin-top:10px">
    <span class="tap-check"><b>✔</b> Evidence-based matching</span>
    <span class="tap-check"><b>✔</b> Source citations</span>
    <span class="tap-check"><b>✔</b> Verified — zero hallucination</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ── Research mode selector — on the home page ────────────────────────────────
_MODES = ["🔍 Prospect Screening", "🔬 Deep Research"]
try:
    mode = st.segmented_control(
        "Research Mode", _MODES, default=_MODES[0],
        key="research_mode", label_visibility="collapsed",
    )
    mode = mode or _MODES[0]
except Exception:  # older Streamlit — fall back to a horizontal radio
    mode = st.radio("Research Mode", _MODES, index=0, horizontal=True,
                    label_visibility="collapsed", key="research_mode_radio")
is_screen = mode.startswith("🔍")

st.caption(
    "Fast 2-source check — QUALIFY / WATCHLIST / SKIP in ~45 seconds. "
    "Use it to triage a long list before deep research."
    if is_screen else
    "Full 6-source investigation — MCA portal, funded partners, CSR decision-makers, "
    "evidence citations, and leadership-ready DOCX report."
)

# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("main_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        company = st.text_input(
            "Company name",
            placeholder="e.g. Ericsson India, Microsoft India, Infosys, Wipro",
            label_visibility="collapsed",
        )
    with col2:
        go = st.form_submit_button(
            "Screen →" if is_screen else "Research →",
            use_container_width=True,
        )

# ── Research ──────────────────────────────────────────────────────────────────
if go and company.strip():
    company = company.strip()
    st.divider()

    bar  = st.progress(0, "Initialising…")
    info = st.empty()

    with st.spinner(f"{'Screening' if is_screen else 'Researching'} **{company}**…"):
        if is_screen:
            info.info("🔎 Checking Sources 1 & 4 (fast mode)…")
            bar.progress(15)
            sources = fetch_screen_sources(company)
            bar.progress(65)
        else:
            steps = ["Searching India CSR page (1/6)…",
                     "Searching MCA portal + CIN (2/6)…",
                     "Searching National CSR Portal (3/6)…",
                     "Searching Annual Report (4/6)…",
                     "Finding funded partners (5/6)…",
                     "Finding CSR decision-makers (6/6)…"]
            step_idx = [0]
            def _cb(msg):
                step_idx[0] = min(step_idx[0] + 1, 5)
                bar.progress(10 + step_idx[0] * 10, steps[min(step_idx[0], 5)])
                info.info(f"🔎 {steps[step_idx[0]]}")
            sources = fetch_deep_sources(company, progress_cb=_cb)
            bar.progress(70)

        info.info("🔬 Parsing & verifying…")
        parsed = parse_all(sources, company)
        bar.progress(85)

        info.info("📊 Scoring…")
        result = compute_score(company, sources, parsed)
        bar.progress(100)

    info.empty()
    bar.empty()

    fit       = result["fit_score"]
    state     = result["state"]
    insight   = result["strategic_insight"]
    breakdown = result.get("breakdown", {})
    data      = result.get("data", {})
    verif     = data.get("verification", {})

    # ════════════════════════════════════════════════════════════════
    # SCREEN MODE output
    # ════════════════════════════════════════════════════════════════
    if is_screen:
        st.markdown(f"## {company}")
        st.markdown(state_badge(state) +
                    delivery_badge(data.get("csr_delivery_model")),
                    unsafe_allow_html=True)
        st.markdown("")

        # STRICT bands: only 80+ goes to the partnership team
        if fit >= 80:
            st.markdown(f"""
            <div class="screen-qualify">
              <div class="screen-pass">✅ QUALIFY — {score_label(fit)}</div>
              <div>Score {fit}/100 — Hand to partnership team. Run Deep Research for the brief.</div>
            </div>""", unsafe_allow_html=True)
        elif fit >= 45:
            st.markdown(f"""
            <div class="screen-review">
              <div class="screen-watchlist">⚠️ {score_label(fit)}</div>
              <div>Score {fit}/100 — Not partnership-ready yet. {"Nurture and strengthen the case." if fit >= 65 else "Revisit quarterly."}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="screen-skip">
              <div class="screen-no">❌ SKIP</div>
              <div>Score {fit}/100 — Low fit with TAP's 21st-century skills mission. Deprioritise.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**Quick signals:**")
            fas = [f.get("value","") for f in data.get("focus_areas",[])]
            if fas:
                for fa in fas[:4]:
                    st.markdown(f"- {fa}")
            adj_fired = breakdown.get("adjacency_boost",{}).get("fired_clusters",[])
            if adj_fired:
                st.markdown("**Adjacent areas:**")
                for c in adj_fired[:3]:
                    st.markdown(f"- {c['label']} (+{c['boost_applied']}pts)")

        st.info("💡 **Tip:** Switch to Deep Research for funded partners, decision-makers, and a leadership-ready DOCX report.")

    # ════════════════════════════════════════════════════════════════
    # DEEP RESEARCH output
    # ════════════════════════════════════════════════════════════════
    else:
        st.markdown(f"## Deep Research: **{company}**")
        st.markdown(state_badge(state) +
                    delivery_badge(data.get("csr_delivery_model")),
                    unsafe_allow_html=True)
        _dm_model = data.get("csr_delivery_model", {})
        if _dm_model.get("model") not in (None, "", "UNCLEAR"):
            with st.expander("Delivery model evidence (funder vs implementer)"):
                st.caption(_dm_model.get("note",""))
                if _dm_model.get("funder_signals"):
                    st.caption("Funder signals: " + ", ".join(_dm_model["funder_signals"]))
                    if _dm_model.get("funder_evidence"):
                        st.caption(f'"{_dm_model["funder_evidence"]}"')
                if _dm_model.get("implementer_signals"):
                    st.caption("Implementer signals: " + ", ".join(_dm_model["implementer_signals"]))
                    if _dm_model.get("implementer_evidence"):
                        st.caption(f'"{_dm_model["implementer_evidence"]}"')
        st.markdown("")

        score_col, insight_col = st.columns([1, 2])
        with score_col:
            c = score_color(fit)
            st.markdown(f"""
            <div class="score-card" style="border:3px solid {c};
                 background:color-mix(in srgb,{c} 8%,transparent)">
              <div class="score-num" style="color:{c}">{fit}</div>
              <div style="font-size:1rem;font-weight:700;color:{c}">{score_label(fit)}</div>
              <div style="font-size:0.78rem;opacity:0.6;margin-top:4px">Fit Score / 100</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Breakdown")
            for dim, info_d in breakdown.items():
                if not isinstance(info_d, dict):
                    continue   # skip "penalties" list and "raw_score" int
                s  = info_d.get("score", 0)
                mx = info_d.get("max", 10)
                st.markdown(f"**{dim_label(dim)}** `{s}/{mx}`")
                st.progress(s / mx if mx else 0)

        with insight_col:
            st.markdown("#### 🧠 Strategic Insight")
            st.markdown(f'<div class="insight-box">{insight}</div>',
                        unsafe_allow_html=True)

            spend = data.get("spend", {})
            if spend.get("inr_crore"):
                st.markdown(f"**💰 India CSR Spend:** `{spend['display']}` {spend.get('usd_approx','')}")
                ef = spend.get("evidence_fact")
                if ef:
                    with st.expander("Spend evidence"):
                        st.caption(f"Source: {ef.get('source_type','')} · {ef.get('source_url','')[:80]}")
                        st.caption(f'"{ef.get("excerpt","")[:280]}"')
            else:
                st.markdown("**💰 India CSR Spend:** Not publicly disclosed (neutral 5 pts)")

            # Verification badge
            vc = verif.get("checks", [])
            if vc:
                st.markdown(
                    f'<span class="tap-pill-green">🛡 Verification: '
                    f'{verif.get("verified",0)}/{len(vc)} facts re-checked against '
                    f'raw sources ({verif.get("pass_rate",100)}%)</span>',
                    unsafe_allow_html=True)

        st.divider()

        # Adjacency clusters
        adj_fired = breakdown.get("adjacency_boost",{}).get("fired_clusters",[])
        if adj_fired:
            st.markdown("### 🔗 Adjacency Clusters")
            st.caption("Areas adjacent to TAP's mission — not exact matches, but strong partnership signals.")
            for c in adj_fired:
                with st.expander(
                    f"**{c['label']}** — +{c['boost_applied']}pts  |  "
                    f"Keywords: {', '.join(c['keywords_found'][:3])}"
                ):
                    st.info(c["tap_reasoning"])
                    for ev in c.get("evidence_excerpts",[])[:1]:
                        st.markdown(f'<div class="ev-box">"{ev[:260]}..."</div>',
                                    unsafe_allow_html=True)

        # ── Funded partners ───────────────────────────────────────────
        st.divider()
        st.markdown("### 🤝 Funded / Implementation Partners")
        st.caption("NGOs this company already funds. **Bold + badge** = similar to TAP "
                   "(known peer NGO or ≥2 education/skilling signals near the mention).")
        partners = data.get("ngo_partners", [])
        if partners:
            for pnr in partners:
                badge = ""
                if pnr.get("is_peer_ngo"):
                    badge = '<span class="tap-pill-green">★ TAP-peer NGO</span>'
                elif pnr.get("tap_similar"):
                    badge = '<span class="tap-pill-green">Similar to TAP</span>'
                own  = ' <span class="tap-pill">own foundation</span>' if pnr.get("is_own_foundation") else ""
                name = f"<b>{pnr['name']}</b>" if pnr.get("tap_similar") else pnr["name"]
                sig  = (f' <span style="opacity:0.6;font-size:0.78rem">· '
                        f'{", ".join(pnr.get("similarity_signals",[])[:4])}</span>'
                        if pnr.get("similarity_signals") else "")
                link = (f' <a href="{pnr["source_url"]}" target="_blank" '
                        f'style="font-size:0.75rem">[source]</a>') if pnr.get("source_url") else ""
                st.markdown(f'<div class="partner-row">{name} {badge}{own}{sig}{link}</div>',
                            unsafe_allow_html=True)
                if pnr.get("excerpt"):
                    with st.expander("evidence", expanded=False):
                        st.caption(f'"{pnr["excerpt"]}"')
        else:
            st.caption("No funded partners found in public sources.")

        # ── Decision makers ───────────────────────────────────────────
        st.markdown("### 👤 CSR Decision-Makers")
        st.caption("Names come only from public sources / LinkedIn's own search snippets — never generated. Verify before outreach.")
        dms = data.get("decision_makers", [])
        if dms:
            for dm in dms:
                li = ""
                if dm.get("linkedin_url"):
                    li = f' · <a href="{dm["linkedin_url"]}" target="_blank">LinkedIn profile</a>'
                elif dm.get("linkedin_search_url"):
                    li = f' · <a href="{dm["linkedin_search_url"]}" target="_blank">search on LinkedIn</a>'
                title = f' — <span style="opacity:0.7">{dm.get("title","")}</span>' if dm.get("title") else ""
                st.markdown(f'<div class="partner-row"><b>{dm.get("name","")}</b>{title}{li}</div>',
                            unsafe_allow_html=True)
                if dm.get("excerpt"):
                    with st.expander("evidence", expanded=False):
                        st.caption(f'"{dm["excerpt"]}"')
        else:
            st.caption("No CSR decision makers identified in public sources.")

        # Data grid
        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("#### 🎯 Focus Areas (Exact Match)")
            for fact in data.get("focus_areas",[]):
                st.markdown(f"- **{fact.get('value','')}**")
                if fact.get("excerpt"):
                    st.caption(f'  _{fact["excerpt"][:160]}…_ — [{fact.get("source_type","")}]({fact.get("source_url","")})')
            if not data.get("focus_areas"):
                st.caption("No exact TAP keywords matched.")

            st.markdown("#### 📋 CSR Maturity")
            sigs = breakdown.get("csr_maturity",{}).get("signals",[])
            if sigs:
                st.markdown(" ".join(f'<span class="tap-pill">{s}</span>' for s in sigs),
                            unsafe_allow_html=True)
            else:
                st.caption("No formal CSR structure signals detected.")

        with right:
            st.markdown("#### 📋 Programmes & Initiatives")
            for prog in data.get("programs",[]):
                st.markdown(f"- {prog.get('name','')}")
            if not data.get("programs"):
                st.caption("No named programmes found.")

            st.markdown("#### 📍 Geography")
            geos = data.get("geography",[])
            if geos:
                st.markdown(" ".join(f'<span class="tap-pill">{g.get("place","")}</span>' for g in geos),
                            unsafe_allow_html=True)
            else:
                st.caption("No India geographies detected.")

        # ── Verification log ──────────────────────────────────────────
        if verif.get("checks"):
            st.divider()
            with st.expander(f"🛡 Verification log — {verif.get('verified',0)}/"
                             f"{len(verif['checks'])} facts verified against raw source text"):
                for c in verif["checks"]:
                    icon = "✅" if c["status"] == "VERIFIED" else "⚠️"
                    st.caption(f"{icon} **{c['field']}** — {c['value']} · {c['status']}")

        # Sources
        st.divider()
        st.markdown("### 🔗 Sources Consulted")
        src_icons = {
            "india_csr_page":      ("1️⃣","Company India CSR Page"),
            "mca_portal":          ("2️⃣","MCA Portal (verified)"),
            "mca_via_search":      ("2️⃣","MCA via Web Search (proxy)"),
            "national_csr_portal": ("3️⃣","National CSR Portal"),
            "annual_report":       ("4️⃣","Annual Report"),
            "global_annual_report":("4️⃣","Annual Report"),
            "partner_search":      ("5️⃣","Funded Partners Search"),
            "people_search":       ("6️⃣","Decision-Makers (LinkedIn)"),
        }
        shown = [s for s in result.get("sources",[]) if s.get("status") != "NOT_TRIED"]
        cols = st.columns(3)
        for i, s in enumerate(shown):
            sn = s.get("source_name","")
            icon, lbl = src_icons.get(sn, ("ℹ️", sn))
            url = s.get("url","")
            with cols[i % 3]:
                if s.get("status") == "FOUND":
                    st.success(f"{icon} **{lbl}**\n\n✅ Found")
                else:
                    st.error(f"{icon} **{lbl}**\n\n⬜ Not found")
                if url:
                    st.caption(f"[View]({url})")

        # Downloads
        st.divider()
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            docx_bytes = generate_docx_report(company, result, mode="deep")
            st.download_button(
                label="📝 Download DOCX (for leadership)",
                data=docx_bytes,
                file_name=f"TAP_CSR_Brief_{company.replace(' ','_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, type="primary",
                help="Leadership-ready Word brief — score visuals, partners, decision-makers, verification log",
            )

        with dl_col2:
            html_report = generate_html_report(company, result, mode="deep")
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name=f"tap_csr_{company.replace(' ','_').lower()}_deep.html",
                mime="text/html",
                use_container_width=True,
                help="Self-contained HTML — can be emailed or opened in any browser",
            )

        with dl_col3:
            export = {
                "company": company, "fit_score": fit, "state": state,
                "strategic_insight": insight, "breakdown": breakdown,
                "csr_data": {
                    "spend": data.get("spend"),
                    "csr_delivery_model": data.get("csr_delivery_model"),
                    "focus_areas": [f.get("value") for f in data.get("focus_areas",[])],
                    "funded_partners": [
                        {"name": n.get("name"), "tap_similar": n.get("tap_similar"),
                         "is_peer_ngo": n.get("is_peer_ngo"),
                         "signals": n.get("similarity_signals"),
                         "source": n.get("source_url")}
                        for n in data.get("ngo_partners",[])],
                    "programs": [p.get("name") for p in data.get("programs",[])],
                    "geography": [g.get("place") for g in data.get("geography",[])],
                    "decision_makers": [
                        {"name": d.get("name"), "title": d.get("title"),
                         "linkedin": d.get("linkedin_url") or d.get("linkedin_search_url"),
                         "source": d.get("source_url")}
                        for d in data.get("decision_makers",[])],
                },
                "verification": verif,
                "sources": [{k:v for k,v in s.items() if k not in ("text","people_hits")}
                            for s in result.get("sources",[])],
            }
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(export, indent=2, ensure_ascii=False, default=str),
                file_name=f"tap_csr_{company.replace(' ','_').lower()}.json",
                mime="application/json",
                use_container_width=True,
            )

elif go:
    st.warning("Please enter a company name.")

# ── How it works (shown before first search) ─────────────────────────────────
if not go:
    st.markdown("")
    st.markdown("##### How it works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("01", "Enter a company", "Type any company with India operations."),
        ("02", "6-source research", "CSR page, MCA, National CSR Portal, annual report, partners, people."),
        ("03", "Evidence & scoring", "Every fact cited, re-verified, and scored across 6 dimensions."),
        ("04", "Share with leadership", "Download a DOCX brief with partners, decision-makers & verification log."),
    ]
    for col, (n, t, d) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f'<div class="step-card"><div class="n">{n}</div>'
                        f'<div class="t">{t}</div><div class="d">{d}</div></div>',
                        unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(
    "TAP CSR Research Agent v5 · fundraising@theapprenticeproject.org · "
    "All data sourced from public web. Every claim carries a source citation "
    "and is re-verified against raw source text. Zero hallucination architecture."
)
