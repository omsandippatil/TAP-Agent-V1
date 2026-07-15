# webui.py — server-rendered HTML for the Flask/Vercel version of the tool.
# Stateless by design: download files are embedded in the results page as
# base64 data URLs, so no session storage is needed between requests
# (Vercel serverless functions share no memory).
import html as _html

from methodology import derive_criteria  # noqa: F401  (imported by api layer)

_CSS = """
:root{--purple:#7C3AED;--green:#16A34A;--blue:#0EA5E9;--amber:#D97706;
      --red:#DC2626;--ink:#1F2937;--muted:#6B7280;--card:#F4EFE8;--line:#E5E7EB}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
     max-width:980px;margin:0 auto;padding:28px 20px;background:#FDFBF7}
h1{font-size:1.9rem;letter-spacing:-0.5px;margin:0}
h1 .accent{color:var(--purple)}
h2{font-size:1.15rem;margin:26px 0 10px;color:var(--purple)}
.sub{color:var(--muted);font-size:0.92rem;margin:6px 0 22px}
form.search{display:flex;gap:10px;margin:18px 0}
input[type=text]{flex:1;padding:12px 14px;font-size:1rem;border:1.5px solid var(--line);
     border-radius:10px}
button{padding:12px 22px;font-size:1rem;font-weight:700;color:#fff;background:var(--purple);
     border:none;border-radius:10px;cursor:pointer}
.modes{display:flex;gap:16px;margin:8px 0;font-size:0.95rem}
.scorecard{display:flex;gap:20px;align-items:stretch;margin:14px 0}
.scorebox{min-width:170px;border-radius:14px;padding:20px;text-align:center;color:#fff}
.scorebox .n{font-size:2.6rem;font-weight:800;line-height:1}
.insight{background:var(--card);border-radius:14px;padding:16px;font-size:0.95rem;flex:1}
.bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:0.85rem}
.bar .lbl{width:190px}
.bar .track{flex:1;height:10px;background:var(--line);border-radius:6px;overflow:hidden}
.bar .fill{height:100%;background:var(--purple)}
table{width:100%;border-collapse:collapse;font-size:0.86rem}
th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#EDE9F8}
.badge{display:inline-block;padding:5px 14px;border-radius:8px;font-weight:700;color:#fff}
.confirm{background:#FFF3CD}
.dl{display:inline-block;margin:8px 10px 0 0;padding:11px 18px;border:1.5px solid var(--purple);
    border-radius:10px;color:var(--purple);font-weight:700;text-decoration:none}
.banner{border-radius:12px;padding:16px 18px;margin:14px 0;font-size:1.02rem}
.small{font-size:0.8rem;color:var(--muted)}
.penalty{color:var(--red);font-size:0.86rem}
a.back{color:var(--muted);font-size:0.9rem}
"""


def _e(s):
    return _html.escape(str(s or ""))


def _page(body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>TAP CSR Research Agent</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")


def _hero() -> str:
    return ("<h1>TAP <span class='accent'>CSR Research Agent</span></h1>"
            "<p class='sub'>Evidence-based corporate funder research for "
            "The Apprentice Project — every claim carries a source citation.</p>")


def render_home(error: str = "") -> str:
    err = f"<p style='color:var(--red)'>{_e(error)}</p>" if error else ""
    body = f"""{_hero()}{err}
<form class="search" method="post" action="/research">
  <input type="text" name="company" placeholder="e.g. Capgemini, HCL Technologies, Bajaj Finserv" required>
  <button type="submit">Research →</button>
</form>
<div class="modes">
  <label><input type="radio" name="mode_pick" value="screen" checked> 🔍 Prospect Screening (~1 min)</label>
  <label><input type="radio" name="mode_pick" value="deep"> 🔬 Deep Research (2–4 min)</label>
</div>
<script>
  // copy the chosen mode into the form as a hidden field
  const form=document.querySelector('form.search');
  const h=document.createElement('input');h.type='hidden';h.name='mode';h.value='screen';
  form.appendChild(h);
  document.querySelectorAll('.modes input').forEach(
    r=>r.addEventListener('change',()=>{h.value=r.value;}));
  form.addEventListener('submit',()=>{const b=form.querySelector('button');
    b.textContent='Researching… please wait';b.disabled=true;});
</script>
<h2>How it works</h2>
<p class="small">Enter a company → the engine researches up to 7 public sources
(India CSR page, MCA CSR-2, National CSR Portal, annual report, funded partners,
decision-makers, announced plans) → every fact is extracted with a verbatim
excerpt → scored 0–100 across 7 dimensions plus the TAP methodology scorecard
(8 criteria, 0–5). Deep Research adds DOCX / HTML / XLSX downloads.</p>"""
    return _page(body)


def _tier_color(result) -> str:
    return (result.get("scoring_tier") or {}).get("color", "#6B7280")


def _bars(breakdown: dict) -> str:
    labels = {"focus_alignment": "🎯 Focus Alignment",
              "adjacency_boost": "🔗 Adjacency Boost",
              "partner_similarity": "🤝 Partner Similarity",
              "geography_fit": "📍 Geography",
              "csr_maturity": "📋 CSR Maturity",
              "budget_size": "💰 Budget",
              "source_quality": "🔍 Source Quality"}
    out = ""
    for dim, info in breakdown.items():
        if not (isinstance(info, dict) and "score" in info and "max" in info):
            continue
        pct = round(100 * info["score"] / info["max"]) if info["max"] else 0
        out += (f"<div class='bar'><span class='lbl'>{labels.get(dim, _e(dim))}</span>"
                f"<span class='track'><span class='fill' style='width:{pct}%'></span></span>"
                f"<b>{info['score']}/{info['max']}</b></div>")
    sem = breakdown.get("semantic_alignment", {})
    if isinstance(sem, dict) and sem.get("used"):
        out += (f"<p class='small'>🤖 AI semantic alignment: "
                f"{_e(sem.get('score'))}/100 — {_e(sem.get('rationale',''))[:200]}</p>")
    return out


def _methodology_table(meth: dict) -> str:
    tier = meth["tier"]
    rows = ""
    for c in meth["criteria"]:
        cls = " class='confirm'" if str(c["evidence"]).startswith("To confirm") else ""
        rows += (f"<tr><td>{_e(c['name'])}</td><td style='text-align:center'>"
                 f"<b>{c['score']}</b></td><td>{_e(c['rating'])}</td>"
                 f"<td{cls}>{_e(c['evidence'])}</td></tr>")
    return (f"<h2>📐 Methodology Scorecard — 8 criteria (0–5)</h2>"
            f"<p><span class='badge' style='background:{tier['color']}'>"
            f"{_e(tier['label'])} · {meth['average']} / 5</span></p>"
            f"<table><tr><th>Criterion</th><th>Score</th><th>Rating</th>"
            f"<th>Evidence (one line)</th></tr>{rows}</table>"
            f"<p class='small'>{_e(meth['csr_head_note'])} Gaps are marked "
            f"'To confirm' — the engine drafts, a person verifies every figure.</p>")


def render_results(company: str, mode: str, result: dict, meth: dict,
                   files: dict) -> str:
    """
    files: {label: (filename, mime, base64_str)} — rendered as data-URL
    download links so the page is fully self-contained (serverless-safe).
    """
    fit   = result.get("fit_score", 0)
    tier  = result.get("scoring_tier", {}) or {}
    color = _tier_color(result)
    bd    = result.get("breakdown", {}) or {}

    if mode == "screen":
        if fit >= 80:
            banner = (f"<div class='banner' style='background:#DCFCE7'>✅ "
                      f"<b>QUALIFY — {fit}/100.</b> Hand to partnership team. "
                      f"Run Deep Research for the full brief.</div>")
        elif fit >= 45:
            banner = (f"<div class='banner' style='background:#FEF3C7'>⚠️ "
                      f"<b>{_e(tier.get('label',''))} — {fit}/100.</b> "
                      f"Not partnership-ready yet.</div>")
        else:
            banner = (f"<div class='banner' style='background:#FEE2E2'>⛔ "
                      f"<b>SKIP — {fit}/100.</b> Low fit — deprioritise.</div>")
    else:
        banner = ""

    penalties = "".join(f"<p class='penalty'>▼ {_e(p.get('reason',''))}</p>"
                        for p in bd.get("penalties", []) or [])

    dls = ""
    if files:
        links = "".join(
            f"<a class='dl' download='{_e(fn)}' href='data:{mime};base64,{b64}'>{_e(lbl)}</a>"
            for lbl, (fn, mime, b64) in files.items())
        dls = (f"<h2>Downloads</h2>{links}"
               f"<p class='small'>Files are embedded in this page — save them "
               f"before closing the tab.</p>")

    body = f"""{_hero()}<a class='back' href='/'>← New search</a>
<h2>{'🔍 Prospect Screening' if mode == 'screen' else '🔬 Deep Research'}: {_e(company)}</h2>
{banner}
<div class="scorecard">
  <div class="scorebox" style="background:{color}">
    <div class="n">{fit}</div><div>{_e(tier.get('label',''))}</div>
    <div class="small" style="color:#fff;opacity:0.75">Fit Score / 100</div>
  </div>
  <div class="insight">{_e(result.get('strategic_insight',''))}</div>
</div>
{penalties}
<h2>Score Breakdown</h2>
{_bars(bd)}
{_methodology_table(meth)}
{dls}
<p class="small">TAP CSR Research Agent · fundraising@theapprenticeproject.org ·
All data from public sources; every claim carries a citation.</p>"""
    return _page(body)
