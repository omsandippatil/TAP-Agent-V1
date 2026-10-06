# FundFinder

> Know who funds the work before you ask them to.

FundFinder researches a company's India CSR activity, scores its fit against your mission, and tells you who to talk to. It reads annual reports so you don't have to.

Scores are opinions with arithmetic. Every claim cites a source. Verify before outreach.

## What it does

| Mode | Purpose | Output |
|---|---|---|
| **Screen** | Fast yes/no on a prospect | Score, tier, summary |
| **Deep Research** | Full dossier | Score, evidence, decision-makers, `.docx` report, `.xlsx` workbook |

## How it works

```
company name
   -> fetch sources (CSR pages, MCA, annual reports, partners, people)
   -> extract facts (LLM)  -> targeted follow-up searches
   -> score against criteria (LLM + deterministic rules)
   -> cited report, saved to history
```

If evidence is thin, the app says so instead of guessing. Silence is not a negative signal.

## Stack

FastAPI, Jinja2 + HTMX, Supabase (Postgres, Storage, Google OAuth), Anthropic API, Google Programmable Search.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # fill it in
uvicorn app.main:app --reload
```

### Environment

| Variable | Notes |
|---|---|
| `SUPABASE_URL`, `SUPABASE_KEY` | Server-side service key |
| `SUPABASE_ANON_KEY` | Enables Google sign-in |
| `SESSION_SECRET` | Required in production |
| `APP_ENV` | `development` or `production` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Default model: `claude-haiku-4-5` |
| `GOOGLE_SEARCH_API_KEY` | Comma-separate multiple keys to rotate |
| `GOOGLE_SEARCH_ENGINE_ID` | Must search the entire web |
| `GOOGLE_SEARCH_DAILY_CAP` | Default `90` |

### Supabase setup

- Tables: `screenings`, `screening_files`, `job_events`, `job_runs`
- Storage bucket: `screening-files`
- Auth: enable Google; add `<your-url>/auth/callback` as a redirect URL

## Configuration

Scoring tiers, bands, partner lists and source toggles live in `config.yaml`.

## Layout

```
app/
  main.py        routes and job runner
  auth.py        Google OAuth, signed sessions
  db.py          Supabase access
  pipeline/      scraping, search budget, LLM scoring
  render/        HTML templates, .docx and .xlsx reports
config.yaml      scoring rules
```

## Deploy

Runs as a **single long-lived process** (jobs are held in memory), so use a container or web service, not serverless. `render.yaml` is included. OCR needs `tesseract` and `poppler` installed.

## Limits

- One worker only.
- Public sources only; some companies simply publish nothing.
- AI-generated analysis. A human checks every figure before it leaves the building.

*Research what can be known. Verify what cannot.*