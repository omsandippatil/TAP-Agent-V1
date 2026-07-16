# TAP CSR Research Agent

FastAPI + Jinja2 + HTMX app, Supabase Postgres backend, deployed on Vercel.

## Local dev
pip install -r requirements.txt
cp .env.example .env   # fill in Supabase credentials
uvicorn app.main:app --reload

## Deploy
vercel
