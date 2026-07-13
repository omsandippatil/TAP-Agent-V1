# TAP CSR Research Agent

An AI-powered research tool that helps [The Apprentice Project (TAP)](https://theapprenticeproject.org) find and prioritize corporate CSR funding prospects in India. Give it a company name and it researches the company's CSR activity across seven sources, scores its fit with TAP's mission on a 0–100 scale, and produces a shareable report for leadership.

## What it does

Type a company name and the agent:

1. **Researches 7 sources** — the company's India CSR page (found via domain discovery, link crawling, and sitemap scanning), MCA portal, the National CSR Portal, annual reports (with PDF extraction), funded/implementation partners, CSR decision-makers on LinkedIn, and recent partnership announcements / future CSR plans / leadership statements.
2. **Extracts evidence** — every fact is cited with a source URL and excerpt, then re-verified.
3. **Scores fit across 6 weighted dimensions** — with an AI semantic layer (Groq / Llama-3.3-70B) that reads the collected evidence against TAP's mission, so companies whose CSR pages say "digital inclusion" instead of "education" still score correctly.
4. **Generates reports** — an interactive dashboard, a downloadable DOCX brief, and an HTML report with partners, decision-makers, spend data, and a verification log.

## Scoring

| Dimension | Weight | What it measures |
|---|---|---|
| Focus Alignment | 55 | How closely CSR focus matches TAP's mission (education-first; AI semantic score can lift this, never lower it) |
| Adjacency Boost | 15 | Related programmes that create partnership openings (government schools, edtech, etc.) |
| Geography Fit | 10 | Presence in TAP's states (Delhi, Maharashtra, etc.) |
| CSR Maturity | 10 | How structured the CSR programme is |
| Budget Size | 5 | India CSR spend (₹ crore tiers) |
| Source Quality | 5 | Strength of the evidence found |

Strict penalties then subtract points for known mismatches (e.g., vocational-training-only or higher-ed-only programmes). Final tiers: **90+ Immediate Target · 80+ Strong Fit · 65+ Conditional · 50+ Watchlist · below 50 Not a Target**.

All weights, focus keywords, adjacency clusters, geography lists, budget tiers, and the mission statement live in `config.yaml` — tune them without touching code.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/RupeshKumar-15/TAP-Agent-V1.git
cd TAP-Agent-V1
pip install -r requirements.txt
```

Enable AI semantic scoring (recommended — without it, scoring falls back to keywords only):

```bash
# 1. Get a free API key at https://console.groq.com
# 2. Copy the template and paste your key in
copy .env.example .env        # Windows  (cp on Mac/Linux)
```

Edit `.env` so it contains:

```
GROQ_API_KEY=gsk_your_real_key_here
```

Verify it's connected:

```bash
python -c "from llm import api_health_check; print(api_health_check())"
```

## Run

```bash
streamlit run app.py
```

Two modes on the home page:

- **🔍 Prospect Screening** — fast first-pass check of a company.
- **🔬 Deep Research** — full 6-source investigation with DOCX + HTML report downloads.

> **Note:** after editing any `.py` file, fully restart Streamlit (Ctrl+C, then `streamlit run app.py`). A browser refresh alone keeps old code cached.

## Project structure

```
app.py            Streamlit UI (dashboard, modes, downloads)
scraper.py        7-source research engine (domain discovery, sitemap scan, PDF extraction)
parser.py         Evidence extraction from fetched text
scorer.py         Weighted 6-dimension scoring + AI semantic lift + strict penalties
llm.py            Groq semantic alignment (silent fallback if no API key)
reporter.py       HTML report generator
docx_reporter.py  DOCX brief generator
config.yaml       All weights, keywords, clusters, tiers, mission text
utils.py          Shared helpers
```

## Known limitations

- **MCA and csr.gov.in block scripted access** (HTTP 403) — spend data comes from annual reports and news sources instead.
- **Semantic scores vary a few points run-to-run** — LLM judgment isn't deterministic; read the AI rationale in the report rather than trusting the number alone.
- **The AI scores the evidence collected, not the company's true activity** — if scraping finds poor sources, the score will be low. Check the verification log when a score looks off.

## For other NGOs

The tool is mission-agnostic: edit `org_mission`, `focus_areas`, and `adjacency_clusters` in `config.yaml` to repurpose it for any nonprofit's corporate fundraising research.
