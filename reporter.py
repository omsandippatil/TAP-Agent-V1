# reporter.py — generates a shareable HTML research brief
"""
Produces a fully self-contained HTML file that:
  - Looks like a professional analyst report
  - Embeds all evidence inline (source citations, excerpts)
  - Can be emailed, saved, or hosted on any web server
  - Zero external dependencies (no CDN links)
  - Supports light AND dark mode (follows the reader's OS preference)
"""

import datetime


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
  :root{
    --bg:#faf7f2; --card:#ffffff; --ink:#1a1a2e; --muted:#6b7280;
    --line:#eae4da; --accent:#7c3aed; --accent-soft:#f3eefe;
    --pill-bg:#ede9fe; --pill-fg:#6d28d9;
    --ev-bg:#fffbeb; --ev-fg:#92400e;
    --cl-bg:#f0fdf4; --cl-line:#bbf7d0; --cl-head:#166534;
    --track:#e9e4db; --shadow:0 2px 14px rgba(26,26,46,.07);
  }
  @media(prefers-color-scheme:dark){
    :root{
      --bg:#12121a; --card:#1c1c28; --ink:#ececf4; --muted:#9ca3af;
      --line:#2c2c3a; --accent:#a78bfa; --accent-soft:#241f38;
      --pill-bg:#2d2547; --pill-fg:#c4b5fd;
      --ev-bg:#2b2415; --ev-fg:#fbbf24;
      --cl-bg:#122318; --cl-line:#1f4d2e; --cl-head:#4ade80;
      --track:#2c2c3a; --shadow:0 2px 14px rgba(0,0,0,.4);
    }
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.65}
  .wrapper{max-width:900px;margin:0 auto;padding:32px 24px}
  header{background:#1a1a2e;color:#fff;padding:28px 32px;border-radius:16px 16px 0 0;
         border-bottom:3px solid var(--accent)}
  header h1{font-size:1.5rem;font-weight:700;letter-spacing:-0.3px}
  header .sub{font-size:0.88rem;opacity:0.7;margin-top:4px;color:#c4b5fd}
  .card{background:var(--card);border-radius:0 0 16px 16px;padding:28px 32px;
        box-shadow:var(--shadow);margin-bottom:24px;border:1px solid var(--line);border-top:none}
  .score-row{display:flex;align-items:center;gap:28px;margin-bottom:24px}
  .score-bubble{min-width:100px;text-align:center;border-radius:16px;
                padding:20px 16px;border:3px solid var(--sc)}
  .score-bubble .num{font-size:3.2rem;font-weight:800;color:var(--sc);line-height:1}
  .score-bubble .lbl{font-size:0.75rem;color:var(--muted);margin-top:4px}
  .score-bubble .fit{font-size:0.9rem;font-weight:700;color:var(--sc)}
  .badge{display:inline-block;padding:4px 14px;border-radius:999px;
         font-size:0.78rem;font-weight:600;color:#fff;margin-bottom:12px}
  .insight-box{background:var(--accent-soft);border-left:4px solid var(--accent);
               padding:16px 18px;border-radius:0 12px 12px 0;
               font-size:0.95rem;line-height:1.7;margin-top:8px}
  h2{font-size:1.05rem;font-weight:700;color:var(--ink);
     border-bottom:2px solid var(--line);padding-bottom:6px;margin:24px 0 14px}
  h3{font-size:0.9rem;font-weight:600;color:var(--ink);margin:14px 0 6px}
  .bar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .bar-label{width:160px;font-size:0.82rem;color:var(--ink);flex-shrink:0}
  .bar-track{flex:1;background:var(--track);border-radius:6px;height:10px}
  .bar-fill{height:10px;border-radius:6px;background:var(--bc)}
  .bar-val{width:44px;font-size:0.8rem;color:var(--muted);text-align:right;flex-shrink:0}
  .cluster{background:var(--cl-bg);border:1px solid var(--cl-line);border-radius:12px;
           padding:12px 14px;margin-bottom:10px}
  .cluster-head{font-weight:600;color:var(--cl-head);font-size:0.88rem}
  .cluster-kws{font-size:0.8rem;color:var(--ink);margin-top:4px}
  .cluster-why{font-size:0.82rem;color:var(--muted);margin-top:6px;line-height:1.5}
  .evidence{background:var(--ev-bg);border-left:3px solid #f59e0b;
            padding:8px 12px;border-radius:0 8px 8px 0;
            font-size:0.78rem;color:var(--ev-fg);margin-top:6px;font-style:italic}
  .evidence a{color:var(--ev-fg)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  ul{list-style:none;padding:0}
  ul li{padding:4px 0;font-size:0.87rem;border-bottom:1px solid var(--line)}
  ul li:last-child{border:none}
  .pill{display:inline-block;background:var(--pill-bg);color:var(--pill-fg);
        border-radius:999px;padding:2px 12px;font-size:0.78rem;margin:2px 3px;font-weight:600}
  .pill-green{display:inline-block;background:var(--cl-bg);color:var(--cl-head);
        border:1px solid var(--cl-line);border-radius:999px;padding:2px 12px;
        font-size:0.75rem;margin:0 4px;font-weight:700}
  .source-row{display:flex;align-items:flex-start;gap:12px;
              padding:10px 0;border-bottom:1px solid var(--line)}
  .source-icon{font-size:1.1rem;flex-shrink:0}
  .source-body{flex:1;font-size:0.83rem}
  .source-body a{color:var(--accent);word-break:break-all}
  a{color:var(--accent)}
  table.vlog{width:100%;border-collapse:collapse;font-size:0.8rem}
  table.vlog th,table.vlog td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
  table.vlog th{background:var(--pill-bg);color:var(--pill-fg)}
  .v-ok{color:var(--cl-head);font-weight:600}
  .v-warn{color:#d97706;font-weight:600}
  .none{color:var(--muted);font-size:0.83rem;font-style:italic}
  footer{text-align:center;font-size:0.75rem;color:var(--muted);margin-top:20px;padding-top:16px;
         border-top:1px solid var(--line)}
  @media(max-width:640px){.grid{grid-template-columns:1fr}.score-row{flex-direction:column}}
"""


def _score_color(s: int) -> str:
    if s >= 90: return "#7c3aed"
    if s >= 80: return "#16a34a"
    if s >= 65: return "#0ea5e9"
    if s >= 45: return "#d97706"
    return "#dc2626"

def _score_label(s: int) -> str:
    if s >= 90: return "PRIORITY MATCH"
    if s >= 80: return "STRONG MATCH"
    if s >= 65: return "PROMISING"
    if s >= 45: return "WATCHLIST"
    return "LOW FIT"

def _delivery_badge(delivery: dict) -> str:
    model = (delivery or {}).get("model", "UNCLEAR")
    m = {"FUNDER":      ("#16a34a", "🤝 FUNDER — grants to NGO partners"),
         "HYBRID":      ("#7c3aed", "🔀 HYBRID — funds partners + runs own programmes"),
         "IMPLEMENTER": ("#d97706", "🏗 IMPLEMENTER — runs CSR in-house"),
         "UNCLEAR":     ("#6b7280", "❔ Delivery model unclear")}
    col, lbl = m.get(model, m["UNCLEAR"])
    return f'<span class="badge" style="background:{col};margin-left:6px">{lbl}</span>'

def _state_badge(state: str) -> str:
    m = {
        "FOUND":               ("#16a34a", "✅ Evidence Found"),
        "NOT_FOUND_IN_SOURCE": ("#d97706", "⚠️ Partial Research"),
        "CONFIRMED_ABSENT":    ("#dc2626", "❌ No CSR Evidence"),
    }
    col, lbl = m.get(state, ("#6b7280", state))
    return f'<span class="badge" style="background:{col}">{lbl}</span>'

def _bar(label: str, score: int, max_score: int, color: str) -> str:
    pct = (score / max_score * 100) if max_score else 0
    return f"""
    <div class="bar-row">
      <span class="bar-label">{label}</span>
      <div class="bar-track" style="--bc:{color}">
        <div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
      </div>
      <span class="bar-val">{score}/{max_score}</span>
    </div>"""

def _evidence_box(excerpt: str, url: str) -> str:
    if not excerpt:
        return ""
    link = f'<br><a href="{url}" target="_blank">→ Source</a>' if url else ""
    return f'<div class="evidence">"{excerpt[:280]}..."{link}</div>'

def _dim_labels():
    return {
        "focus_alignment": ("🎯 Focus Alignment", "#7c3aed"),
        "adjacency_boost": ("🔗 Adjacency Boost", "#8b5cf6"),
        "partner_similarity": ("🤝 Partner Similarity", "#0ea5e9"),
        "geography_fit":   ("📍 Geography",       "#f59e0b"),
        "csr_maturity":    ("📋 CSR Maturity",    "#10b981"),
        "budget_size":     ("💰 Budget",           "#ef4444"),
        "source_quality":  ("🔍 Source Quality",  "#6b7280"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(company: str, result: dict, mode: str = "deep") -> str:
    """
    Returns a complete, self-contained HTML string.
    Save it as a .html file — it works with no internet connection.
    """
    fit       = result["fit_score"]
    state     = result["state"]
    insight   = result["strategic_insight"]
    breakdown = result.get("breakdown", {})
    data      = result.get("data", {})
    sources   = result.get("sources", [])
    sc_color  = _score_color(fit)
    now       = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Score breakdown bars ──────────────────────────────────────────────────
    dim_map = _dim_labels()
    bars_html = ""
    for dim, info in breakdown.items():
        if not (isinstance(info, dict) and "score" in info and "max" in info):
            continue   # skip penalties (list), raw_score (int), semantic_alignment
        lbl, col = dim_map.get(dim, (dim, "#6b7280"))
        bars_html += _bar(lbl, info.get("score",0), info.get("max",10), col)

    sem_bd = breakdown.get("semantic_alignment", {})
    if isinstance(sem_bd, dict) and sem_bd.get("used"):
        bars_html += _bar("AI Semantic Alignment",
                          sem_bd.get("score") or 0, 100, "#7c3aed")

    # ── Methodology scorecard (8 criteria, 0–5) ───────────────────────────────
    meth_html = ""
    try:
        from methodology import derive_criteria
        from scorer import _cfg as _load_cfg
        _meth = derive_criteria(company, result, _load_cfg())
    except Exception:
        _meth = None   # additive — never block the report
    if _meth:
        _tier = _meth["tier"]
        _rows = ""
        for c in _meth["criteria"]:
            _ev_style = ("background:#FFF3CD;" if
                         c["evidence"].startswith("To confirm") else "")
            _rows += (f"<tr><td>{c['name']}</td>"
                      f"<td style='text-align:center'><b>{c['score']}</b></td>"
                      f"<td>{c['rating']}</td>"
                      f"<td style='{_ev_style}color:var(--muted)'>{c['evidence']}</td></tr>")
        _oq = " · ".join(_meth.get("open_questions", [])[:4])
        meth_html = f"""
  <h2>📐 Methodology Scorecard — 8 Criteria (0–5)</h2>
  <p><span style="display:inline-block;padding:5px 14px;border-radius:8px;
     font-weight:700;color:#fff;background:{_tier['color']}">
     {_tier['label']} · {_meth['average']} / 5</span></p>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
    <tr style="background:#EDE9F8">
      <th align="left" style="padding:6px">Criterion</th>
      <th style="padding:6px">Score</th>
      <th align="left" style="padding:6px">Rating</th>
      <th align="left" style="padding:6px">Evidence (one line)</th></tr>
    {_rows}
  </table>
  <p style="font-size:0.8rem;color:var(--muted)">{_meth['csr_head_note']}
     Gaps are marked 'To confirm' — the engine drafts, a person verifies
     every figure. Open questions: {_oq}</p>"""

    # ── Adjacency clusters ────────────────────────────────────────────────────
    adj_html = ""
    adj_fired = breakdown.get("adjacency_boost",{}).get("fired_clusters",[])
    for c in adj_fired:
        kws = ", ".join(c.get("keywords_found",[])[:4])
        excerpts_html = ""
        for ex in c.get("evidence_excerpts",[])[:1]:
            excerpts_html += f'<div class="evidence">"{ex[:240]}..."</div>'
        adj_html += f"""
        <div class="cluster">
          <div class="cluster-head">{c['label']} &nbsp; <span style="color:#059669">+{c['boost_applied']}pts</span></div>
          <div class="cluster-kws">Keywords found: <b>{kws}</b></div>
          <div class="cluster-why">{c.get('tap_reasoning','')}</div>
          {excerpts_html}
        </div>"""

    if not adj_html:
        adj_html = '<p class="none">No adjacency clusters fired.</p>'

    # ── Focus areas ───────────────────────────────────────────────────────────
    fa_items = ""
    for fact in data.get("focus_areas",[]):
        ev = _evidence_box(fact.get("excerpt",""), fact.get("source_url",""))
        fa_items += f"<li><b>{fact.get('value','')}</b>{ev}</li>"
    if not fa_items:
        fa_items = '<li class="none">No exact TAP focus area keywords matched.</li>'

    # ── Funded partners (with TAP-similarity) ─────────────────────────────────
    ngo_items = ""
    for ngo in data.get("ngo_partners",[]):
        url   = ngo.get("source_url","")
        link  = f' <a href="{url}" target="_blank" style="font-size:0.75rem">[source]</a>' if url else ""
        badge = ""
        if ngo.get("is_peer_ngo"):
            badge = '<span class="pill-green">★ TAP-peer NGO</span>'
        elif ngo.get("tap_similar"):
            badge = '<span class="pill-green">Similar to TAP</span>'
        own  = '<span class="pill">own foundation</span>' if ngo.get("is_own_foundation") else ""
        sig  = (f'<span style="font-size:0.75rem;color:var(--muted)"> · '
                f'{", ".join(ngo.get("similarity_signals",[])[:4])}</span>'
                if ngo.get("similarity_signals") else "")
        name = f"<b>{ngo.get('name','')}</b>" if ngo.get("tap_similar") else ngo.get("name","")
        ngo_items += f"<li>{name} {badge}{own}{sig}{link}</li>"
    if not ngo_items:
        ngo_items = '<li class="none">No funded partners found in sources.</li>'

    # ── Programs ──────────────────────────────────────────────────────────────
    prog_items = ""
    for prog in data.get("programs",[]):
        prog_items += f"<li>{prog.get('name','')}</li>"
    if not prog_items:
        prog_items = '<li class="none">No named programmes found.</li>'

    # ── Geography ─────────────────────────────────────────────────────────────
    geo_pills = "".join(
        f'<span class="pill">{g.get("place","")}</span>'
        for g in data.get("geography",[])
    ) or '<span class="none">No specific India geographies detected.</span>'

    # ── Decision makers (with LinkedIn) ───────────────────────────────────────
    dm_items = ""
    for dm in data.get("decision_makers",[]):
        ev = _evidence_box(dm.get("excerpt",""), dm.get("source_url",""))
        title = f' — <span style="color:var(--muted)">{dm.get("title","")}</span>' if dm.get("title") else ""
        li = ""
        if dm.get("linkedin_url"):
            li = f' · <a href="{dm["linkedin_url"]}" target="_blank" style="font-size:0.78rem">LinkedIn profile</a>'
        elif dm.get("linkedin_search_url"):
            li = f' · <a href="{dm["linkedin_search_url"]}" target="_blank" style="font-size:0.78rem">search on LinkedIn</a>'
        dm_items += f"<li><b>{dm.get('name','')}</b>{title}{li}{ev}</li>"
    if not dm_items:
        dm_items = '<li class="none">No CSR decision makers identified.</li>'

    # ── CSR Spend ─────────────────────────────────────────────────────────────
    spend     = data.get("spend", {})
    spend_val = spend.get("display","Not publicly disclosed")
    spend_usd = spend.get("usd_approx","")
    spend_ev  = ""
    ef = spend.get("evidence_fact")
    if ef:
        spend_ev = _evidence_box(ef.get("excerpt",""), ef.get("source_url",""))

    # ── Sources ───────────────────────────────────────────────────────────────
    src_icons = {
        "india_csr_page":      ("1️⃣", "Company India CSR Page"),
        "mca_portal":          ("2️⃣", "MCA Portal (verified)"),
        "mca_via_search":      ("2️⃣", "MCA via Web Search (proxy — not a verified filing)"),
        "national_csr_portal": ("3️⃣", "National CSR Portal"),
        "annual_report":       ("4️⃣", "Annual / Sustainability Report"),
        "global_annual_report":("4️⃣", "Annual / Sustainability Report"),
        "partner_search":      ("5️⃣", "Funded Partners Search"),
        "people_search":       ("6️⃣", "Decision-Makers (LinkedIn snippets)"),
        "plans_search":        ("7️⃣", "Partnerships & Announced Plans"),
    }
    src_rows = ""
    for s in sources:
        sn  = s.get("source_name","")
        icon, lbl = src_icons.get(sn, ("ℹ️", sn))
        status = s.get("status","")
        url    = s.get("url","")
        if status == "NOT_TRIED":
            continue
        status_html = (
            '<span style="color:#16a34a;font-weight:600">✅ Found</span>'
            if status == "FOUND" else
            '<span style="color:#9ca3af">⬜ Not found</span>'
        )
        url_html = f'<a href="{url}" target="_blank">{url[:70]}{"…" if len(url)>70 else ""}</a>' if url else "—"
        src_rows += f"""
        <div class="source-row">
          <span class="source-icon">{icon}</span>
          <div class="source-body">
            <b>{lbl}</b> &nbsp; {status_html}<br>{url_html}
          </div>
        </div>"""

    # ── Verification log ──────────────────────────────────────────────────────
    verif = data.get("verification", {})
    verif_html = ""
    checks = verif.get("checks", [])
    if checks:
        rows = "".join(
            f'<tr><td>{c["field"]}</td><td>{c["value"]}</td>'
            f'<td class="{"v-ok" if c["status"]=="VERIFIED" else "v-warn"}">{c["status"]}</td></tr>'
            for c in checks[:30])
        verif_html = f"""
  <h2>🛡 Verification Log</h2>
  <p style="font-size:0.82rem;color:var(--muted);margin-bottom:10px">
    Every published fact was re-checked against raw source text:
    <b>{verif.get("verified",0)}/{len(checks)} verified ({verif.get("pass_rate",100)}%)</b>.
    Facts without source evidence are never shown.
  </p>
  <table class="vlog"><tr><th>Field</th><th>Value</th><th>Status</th></tr>{rows}</table>"""

    # ── CSR Maturity signals ──────────────────────────────────────────────────
    mat_sigs = breakdown.get("csr_maturity",{}).get("signals",[])
    mat_html = "".join(f'<span class="pill">{s}</span>' for s in mat_sigs) \
               or '<span class="none">No formal CSR structure signals detected.</span>'

    mode_label = "Prospect Screening" if mode == "screen" else "Deep Research"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAP CSR Research — {company}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrapper">

<header>
  <div class="sub">The Apprentice Project · CSR Research Agent · {mode_label} · {now}</div>
  <h1>CSR Research Brief: {company}</h1>
</header>

<div class="card">

  <!-- Score + insight -->
  <div class="score-row">
    <div class="score-bubble" style="--sc:{sc_color};background:{sc_color}12">
      <div class="num">{fit}</div>
      <div class="fit">{_score_label(fit)}</div>
      <div class="lbl">Fit Score / 100</div>
    </div>
    <div style="flex:1">
      {_state_badge(state)}{_delivery_badge(data.get("csr_delivery_model"))}
      <div class="insight-box">{insight}</div>
    </div>
  </div>

  <!-- Score breakdown -->
  <h2>Score Breakdown</h2>
  {bars_html}

  <!-- Methodology scorecard (8 criteria, 0-5) -->
  {meth_html}

  <!-- CSR Spend -->
  <h2>💰 India CSR Spend</h2>
  <p style="font-size:0.95rem"><b>{spend_val}</b>
    {"&nbsp; " + spend_usd if spend_usd else ""}
  </p>
  {spend_ev}

  <!-- Adjacency clusters -->
  <h2>🔗 Adjacency Clusters</h2>
  <p style="font-size:0.82rem;color:var(--muted);margin-bottom:12px">
    Areas where the company invests that are <em>adjacent</em> to TAP's mission —
    not exact matches, but strong partnership signals.
  </p>
  {adj_html}

  <!-- Data grid -->
  <div class="grid" style="margin-top:24px">
    <div>
      <h2>🎯 Focus Areas (Exact)</h2>
      <ul>{fa_items}</ul>

      <h2>🤝 Funded / Implementation Partners</h2>
      <ul>{ngo_items}</ul>

      <h2>📋 CSR Maturity Signals</h2>
      <div>{mat_html}</div>
    </div>
    <div>
      <h2>📋 Programmes & Initiatives</h2>
      <ul>{prog_items}</ul>

      <h2>👤 CSR Decision-Makers</h2>
      <ul>{dm_items}</ul>

      <h2>📍 Geography</h2>
      <div>{geo_pills}</div>
    </div>
  </div>

  {verif_html}

  <!-- Sources -->
  <h2>🔗 Sources Consulted</h2>
  {src_rows}

</div><!-- /card -->

<footer>
  Generated by TAP CSR Research Agent v5 · fundraising@theapprenticeproject.org<br>
  All data sourced from public web. Every claim carries a source citation.
  This report is for internal use only.
</footer>

</div><!-- /wrapper -->
</body>
</html>"""

    return html


def save_html_report(company: str, result: dict, mode: str = "deep",
                     output_dir: str = ".") -> str:
    """Save HTML report to disk. Returns the file path."""
    import os
    html  = generate_html_report(company, result, mode)
    slug  = company.lower().replace(" ","_").replace("/","_")
    fname = f"tap_csr_{slug}_{mode}.html"
    path  = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
