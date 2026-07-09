# 🎯 TAP CSR Research Agent

**An automated pre-sales research tool for The Apprentice Project (TAP).** Type a company name, and in under a minute the agent researches the company's India CSR activity across six public sources, scores how well it fits TAP as a funding partner, and produces a leadership-ready report — with every claim backed by a cited, re-verified quote from a public source.

> **TAP** (The Apprentice Project) builds 21st-century skills — critical thinking, creativity, confidence, communication, financial literacy — in low-income middle- and high-school students across India, delivered through **TAP Buddy**, an AI/NLP-powered WhatsApp chatbot.

---

## The problem it solves

Fundraising teams spend hours manually researching a single company: finding its CSR page, MCA filings, annual report, which NGOs it already funds, and who runs its CSR programme. The research is slow, inconsistent between researchers, and hard to audit.

This agent does all of it automatically, applies TAP's partnership criteria consistently every time, and outputs a defensible verdict: **QUALIFY, WATCHLIST, or SKIP**.

## What it does

- Researches any company with India operations across **6 public sources**
- Extracts CSR spend, focus areas, funded NGO partners, programmes, geography, and CSR decision-makers
- Detects whether the company is a **FUNDER** (grants to NGOs — ideal for TAP), an **IMPLEMENTER** (runs CSR in-house), or a **HYBRID**
- Scores partnership fit **out of 100** across six weighted dimensions
- Attaches a **verbatim source quote** to every fact and **re-verifies** each quote against the raw fetched text
- Exports a **DOCX leadership brief**, a **self-contained HTML report**, and **JSON**

## Two research modes

| Mode | Sources | Time | Use case |
|---|---|---|---|
| 🔍 **Prospect Screening** | 1 & 4 only | ~45 s | Triage a long list fast — binary qualify / don't qualify |
| 🔬 **Deep Research** | All 6 | ~2–4 min | Full dossier: partners, decision-makers, verification log, DOCX brief |

---

## How it works — the pipeline

```
 User input (company name + mode)          app.py
        │
        ▼
 1. SCRAPE — 6 public sources              scraper.py
    S1  Company India CSR page      (direct URL guesses → search fallback)
    S2  MCA portal                  (CIN lookup via regex → master data → search proxy)
    S3  National CSR Portal         (csr.gov.in direct → site: search)
    S4  Annual report               (search → PDF parsing via pdfplumber)
    S5  Funded / implementation partners  (targeted queries)
    S6  CSR decision-makers         (LinkedIn search snippets only — never fabricated)
    Every page must pass relevance guards: mentions the company,
    contains CSR language, and meets a minimum length — or it is NOT_FOUND.
        │
        ▼
 2. PARSE — 8 extractors                   parser.py
    spend (₹ crore/lakh regex + false-positive filter) · focus areas
    (30 weighted TAP keywords) · adjacency signals (8 clusters) ·
    NGO partners (peer list + pattern matching) · programmes ·
    geography · decision-makers · funder-vs-implementer model
    → every fact becomes an EvidencedFact {value, confidence,
      source_url, verbatim excerpt} — no source, no fact.
        │
        ▼
 3. VERIFY — anti-hallucination pass       parser.py :: verify_facts()
    Each excerpt is re-checked to literally exist in the raw fetched
    text → VERIFIED / CHECK MANUALLY + pass-rate, shown as a log.
        │
        ▼
 4. SCORE — 6 weighted dimensions          scorer.py
    Focus alignment 40 · Adjacency 20 · Geography 10 ·
    CSR maturity 10 · Budget 10 · Source quality 10  →  fit /100
    Bands: 90+ Priority · 80+ Strong · 65+ Promising ·
    45+ Watchlist · <45 Low fit
        │
        ▼
 5. RENDER                                 app.py / reporter.py / docx_reporter.py
    Streamlit dashboard · DOCX brief · HTML report · JSON export
```

The engine is **fully deterministic** — keyword, regex, and rule-based. The same input always produces the same output, every score is auditable down to the quoted evidence, and it costs nothing to run.

### The zero-hallucination design

Two mechanisms make it structurally impossible for the tool to invent facts:

1. **`EvidencedFact` contract** (`utils.py`) — every extracted data point must carry a source URL and a verbatim excerpt. Facts that fail `is_verified()` are dropped before display.
2. **Verification pass** (`parser.py`) — before publishing, every excerpt is re-checked to literally exist in the fetched source text. Failures are flagged "CHECK MANUALLY" in a visible log.

---

## File-by-file architecture

| File | Role | What it does |
|---|---|---|
| `app.py` | The face | Streamlit UI: mode selector, input form, progress, results dashboard, download buttons. Orchestrates the pipeline; contains no research logic. |
| `scraper.py` | The hands | Fetches the 6 sources: DuckDuckGo search (`ddgs`), direct URL guessing, `requests` + BeautifulSoup for HTML, `pdfplumber` for annual-report PDFs, relevance guards. |
| `parser.py` | The brain (reading) | 8 extractors producing `EvidencedFact`s, plus `verify_facts()` — the anti-hallucination double-check. |
| `scorer.py` | The brain (judging) | 6-dimension weighted scoring, verdict bands, templated strategic-insight paragraph. |
| `config.yaml` | The knowledge | All domain expertise as data: weighted focus keywords, 8 adjacency clusters with reasoning, 43 peer-NGO list, budget tiers, funder/implementer signals, false-positive filters. Retune the tool without touching code. |
| `utils.py` | The contract | `EvidencedFact` dataclass, shared HTTP session with retries, text/excerpt helpers. |
| `reporter.py` | Output | Self-contained HTML report generator. |
| `docx_reporter.py` | Output | Leadership-ready Word brief (python-docx): score visuals, partners, decision-makers, verification log. |

### `config.yaml` — the intelligence layer

The scoring logic is **data, not code**. Highlights:

- **`tap_focus_areas`** — 30 keywords weighted 40–100 by closeness to TAP's mission ("21st century skills" = 100, "education" = 55)
- **`adjacency_clusters`** — the core intelligence: 8 clusters (government schools, digital education, SEL, financial literacy, STEM, learning quality, teacher training, employability) that catch companies funding *adjacent* work even when exact keywords don't match, each with written reasoning for why it matters to TAP
- **`tap_peer_ngos`** — 43 known TAP-like NGOs; a company already funding one is the strongest possible signal
- **Strict focus scoring** — one generic keyword can't turn a company green: single matches are penalised ×0.6 and generic-only matches hard-cap at 12/40

---

## Getting started

### Prerequisites

- Python 3.10+ (developed on 3.12)
- Internet connection (the agent fetches live public web data)

### Install & run

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Type a company name (e.g. *Infosys*, *Ericsson India*), pick a mode, and hit **Screen →** or **Research →**.

### Dependencies

| Package | Used for |
|---|---|
| `streamlit` | Web UI |
| `ddgs` | DuckDuckGo search (no API key required) |
| `requests`, `beautifulsoup4`, `lxml` | Fetching & parsing web pages |
| `pdfplumber` | Extracting text from annual-report PDFs |
| `python-docx` | DOCX brief generation |
| `pyyaml` | Loading `config.yaml` |

**No API keys are required to run the current version.** All data comes from public web pages and free search.

---

## 🤖 Anthropic API (Claude) — planned AI layer

### Current status

The current version is **100% deterministic — it makes no AI calls and needs no API key.** This is a deliberate first-version choice: it guarantees zero hallucination, full auditability, and zero running cost.

### Why add Claude, then?

Keyword matching has a ceiling. It misses paraphrases — a company that writes *"we nurture the problem-solvers of tomorrow"* scores zero for *critical thinking* because the exact keyword never appears. Regex-extracted partner and people names are noisy. The strategic-insight text is templated and stiff.

Claude adds **semantic understanding** at exactly the points where rules fall short, while the existing verification layer keeps it honest:

| Integration point | File | What changes |
|---|---|---|
| **Semantic extraction** | `parser.py` | Claude reads the raw source text and extracts focus areas, partners, decision-makers, spend and delivery model — including paraphrased mentions the keywords miss. It must return a **verbatim quote** for every item; quotes are then run through the existing `verify_facts()` literal-match check, so anything hallucinated is automatically rejected. *Claude proposes, the verifier disposes.* |
| **Source triage** | `scraper.py` | A fast Claude Haiku call answers "is this page really about *this* company's CSR in India?" — far more reliable than token-overlap heuristics at rejecting wrong-company pages. |
| **Strategic insight** | `scorer.py` / `reporter.py` | Claude writes a tailored partnership pitch from the *verified* facts (not raw text) — replacing templated sentences with a genuinely useful outreach angle. |

What stays deterministic on purpose: the **scoring arithmetic** (auditable, reproducible, tunable via `config.yaml`) and the **verification pass** (the trust anchor must remain literal string matching).

### What the key changes in practice

| | Without API key (current) | With `ANTHROPIC_API_KEY` |
|---|---|---|
| Extraction | Exact keyword/regex only | Semantic — catches paraphrases & context |
| Partner/people names | Noisy regex candidates | Clean, contextual extraction |
| Insight text | Templated sentences | Tailored pitch written from verified facts |
| Hallucination risk | Zero (nothing generative) | Still ~zero — every AI claim must survive the literal-quote verifier |
| Cost per deep-research run | Free | A few cents (Haiku for triage, Sonnet for extraction) |
| Reproducibility | Identical output every run | Extraction may vary slightly (temperature 0 minimises this) |

### Setup (once implemented)

```bash
# Get a key at https://console.anthropic.com
# Windows:
set ANTHROPIC_API_KEY=sk-ant-...
# macOS/Linux:
export ANTHROPIC_API_KEY=sk-ant-...
```

The design is **graceful-fallback**: with no key set, the app runs exactly as today on the keyword engine. Never commit your key — keep it in an environment variable or a git-ignored `.env` file.

---

## Limitations

- **Keyword ceiling** — deterministic parsing misses paraphrased CSR language (the trade-off that buys zero hallucination; the planned Claude layer addresses it).
- **Public data only** — companies that don't publish CSR details score low on evidence, not on actual activity. `CONFIRMED_ABSENT` means "nothing public found", not "no CSR".
- **Source availability** — MCA and csr.gov.in are JS/CAPTCHA-gated; the agent falls back to search-snippet proxies and honestly down-weights them in `source_quality`.
- **Rate limits** — DuckDuckGo may throttle rapid consecutive searches; deep runs are sequential with small delays.

