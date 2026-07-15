# api/index.py — Vercel serverless entrypoint (Flask)
# All routes are rewritten here by vercel.json. The research pipeline modules
# live in the repo root, one directory up.
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request  # noqa: E402

from scraper import fetch_screen_sources, fetch_deep_sources        # noqa: E402
from parser import parse_all                                        # noqa: E402
from scorer import score as compute_score, _cfg as load_cfg         # noqa: E402
from methodology import derive_criteria                             # noqa: E402
from reporter import generate_html_report                           # noqa: E402
from docx_reporter import generate_docx_report                      # noqa: E402
from deep_dive_xlsx import generate_deep_dive_xlsx                  # noqa: E402
from webui import render_home, render_results                       # noqa: E402

app = Flask(__name__)

_DOCX_MIME = ("application/vnd.openxmlformats-officedocument."
              "wordprocessingml.document")
_XLSX_MIME = ("application/vnd.openxmlformats-officedocument."
              "spreadsheetml.sheet")


@app.get("/")
def home():
    return render_home()


@app.post("/research")
def research():
    company = (request.form.get("company") or "").strip()
    mode    = request.form.get("mode", "screen")
    if mode not in ("screen", "deep"):
        mode = "screen"
    if not company:
        return render_home(error="Please enter a company name.")

    # ── Pipeline: fetch → parse → score → methodology ────────────────────────
    sources = (fetch_screen_sources(company) if mode == "screen"
               else fetch_deep_sources(company))
    parsed  = parse_all(sources, company)
    result  = compute_score(company, sources, parsed)
    cfg     = load_cfg()
    meth    = derive_criteria(company, result, cfg)

    # ── Deep mode: generate all report files now and embed them in the page
    #    (serverless functions share no memory between requests, so there is
    #    no session to fetch them from later) ──────────────────────────────────
    files = {}
    if mode == "deep":
        safe = company.replace(" ", "_")

        docx_bytes = generate_docx_report(company, result, mode="deep")
        files["📝 DOCX brief (leadership)"] = (
            f"TAP_CSR_Brief_{safe}.docx", _DOCX_MIME,
            base64.b64encode(docx_bytes).decode())

        html_report = generate_html_report(company, result, mode="deep")
        files["📄 HTML report"] = (
            f"tap_csr_{safe.lower()}_deep.html", "text/html",
            base64.b64encode(html_report.encode("utf-8")).decode())

        xlsx_bytes = generate_deep_dive_xlsx(company, result, cfg)
        files["📊 Deep-dive base (XLSX, 7 sheets)"] = (
            f"TAP_DeepDive_{safe}.xlsx", _XLSX_MIME,
            base64.b64encode(xlsx_bytes).decode())

        export = dict(result)
        export["sources"] = [
            {k: v for k, v in s.items() if k not in ("text", "people_hits")}
            for s in result.get("sources", [])]
        export["methodology"] = meth
        files["⬇️ JSON export"] = (
            f"tap_csr_{safe.lower()}.json", "application/json",
            base64.b64encode(json.dumps(
                export, indent=2, ensure_ascii=False, default=str
            ).encode("utf-8")).decode())

    return render_results(company, mode, result, meth, files)


# Local development: `python api/index.py` then open http://localhost:5000
if __name__ == "__main__":
    app.run(debug=True)
