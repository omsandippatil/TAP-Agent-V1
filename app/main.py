import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import auth, db
from app.pipeline import scorer, scraper
from app.pipeline.config_loader import load_config
from app.pipeline.source_registry import SourceRegistry
from app.render.xlsx_reporter import generate_deep_dive_xlsx
from app.render.docx_reporter import generate_docx_report
from app.render.reporters import render_results, render_results_page, templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("tap.main")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

JOBS: dict[str, dict] = {}
JOB_TTL_SECONDS = 60 * 30
DELIVERED_JOB_TTL_SECONDS = 60 * 5

HISTORY_SEARCH_RESULT_LIMIT = 8
FRESH_RESULT_MAX_AGE = {"screen": timedelta(hours=6), "deep": timedelta(days=3)}

_INFLIGHT_JOB_BY_KEY: dict[tuple, str] = {}
_JOBS_LOCK = asyncio.Lock()


def _is_htmx_request(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


def _encode_file(content_bytes: bytes, filename: str, mime: str) -> tuple[str, str, str]:
    b64 = base64.b64encode(content_bytes).decode("ascii")
    return filename, mime, b64


def _slugify(company: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in company.strip().lower()).strip("_") or "company"


def _job_key(company: str, mode: str) -> tuple:
    return (_slugify(company), mode)


def _job_ttl_seconds(job: dict) -> int:
    if job.get("status") in ("delivered", "delivered_error"):
        return DELIVERED_JOB_TTL_SECONDS
    return JOB_TTL_SECONDS


def _prune_expired_jobs():
    now = time.monotonic()
    expired = [
        job_id for job_id, job in JOBS.items()
        if job["status"] != "running" and now - job["created_at"] > _job_ttl_seconds(job)
    ]
    for job_id in expired:
        job = JOBS.pop(job_id, None)
        if job:
            key = _job_key(job["company"], job["mode"])
            if _INFLIGHT_JOB_BY_KEY.get(key) == job_id:
                _INFLIGHT_JOB_BY_KEY.pop(key, None)


def _current_user(request: Request) -> dict | None:
    if not auth.auth_configured():
        return None
    return auth.get_current_user(request)


def _redirect_uri(request: Request) -> str:
    return str(request.base_url).rstrip("/") + "/auth/callback"


async def _complete_oauth_login(code: str) -> tuple[RedirectResponse | None, str | None]:
    session_payload, error = auth.exchange_code_for_session(code)
    if error:
        return None, error
    response = RedirectResponse("/", status_code=302)
    auth.set_session_cookie(response, session_payload)
    return response, None


def _rows_to_history_results(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row.get("id"),
            "company": row.get("company"),
            "logo_url": row.get("logo_url"),
            "fit_score": row.get("fit_score"),
            "tier_label": row.get("tier_label"),
            "tier_color": row.get("tier_color"),
            "mode": row.get("mode"),
            "created_at": row.get("created_at"),
            "source": "history",
        }
        for row in rows
    ]


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _is_fresh_enough(row: dict, mode: str) -> bool:
    created_at = _parse_created_at(row.get("created_at"))
    if created_at is None:
        return False
    max_age = FRESH_RESULT_MAX_AGE.get(mode, FRESH_RESULT_MAX_AGE["screen"])
    return datetime.now(timezone.utc) - created_at <= max_age


def _find_reusable_screening(company: str, mode: str) -> dict | None:
    rows = db.company_history(company, limit=10)
    for row in rows:
        if row.get("mode") == mode and _is_fresh_enough(row, mode):
            return row
    if mode == "screen":
        for row in rows:
            if row.get("mode") == "deep" and _is_fresh_enough(row, "deep"):
                return row
    return None


async def _run_screen_job(job_id: str, company: str, mode: str, user_id: str | None):
    job = JOBS.get(job_id)
    if job is None:
        logger.warning("job MISSING at start job_id=%s company=%r mode=%s", job_id, company, mode)
        return

    key = _job_key(company, mode)
    try:
        request_started_at = time.monotonic()
        logger.info("job START job_id=%s company=%r mode=%s", job_id, company, mode)
        db.log_job_run(company, mode, "start", f"job_id={job_id}")

        cfg = load_config()
        search_cfg = cfg.get("search_source_toggles", {})
        registry = SourceRegistry(company)

        if mode == "deep":
            sources = await scraper.fetch_deep_sources(company, search_cfg, registry=registry)
        else:
            sources = await scraper.fetch_screen_sources(company, search_cfg, registry=registry)

        found_count = sum(1 for s in sources if s.get("status") == "FOUND")
        logger.info("job sources job_id=%s company=%r found=%d/%d", job_id, company, found_count, len(sources))
        db.log_job_run(company, mode, "sources_fetched", f"found={found_count}/{len(sources)}")

        result = await scorer.score(company, sources, cfg, registry=registry)
        logger.info(
            "job scored job_id=%s company=%r state=%s fit_score=%s analysis_present=%s source_bank_size=%d",
            job_id, company, result.get("state"), result.get("fit_score"), bool(result.get("analysis")),
            len(result.get("source_bank", [])),
        )

        files = {}
        docx_filename = None
        xlsx_filename = None
        docx_bytes = None
        xlsx_bytes = None
        if mode == "deep":
            slug = _slugify(company)
            docx_bytes = await generate_docx_report(company, result, mode)
            docx_filename = f"tap_csr_{slug}_deep.docx"
            files["Word Report (.docx)"] = _encode_file(
                docx_bytes,
                docx_filename,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            xlsx_bytes = await generate_deep_dive_xlsx(company, result, cfg)
            xlsx_filename = f"tap_csr_{slug}_deep_dive.xlsx"
            files["Deep-Dive Workbook (.xlsx)"] = _encode_file(
                xlsx_bytes,
                xlsx_filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        screening_row = db.save_screening_result(company, mode, result, cfg, user_id=user_id)
        screening_id = screening_row["id"] if screening_row else None

        if screening_id and mode == "deep":
            db.save_screening_file(
                screening_id, "Word Report (.docx)", docx_filename,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document", docx_bytes,
            )
            db.save_screening_file(
                screening_id, "Deep-Dive Workbook (.xlsx)", xlsx_filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", xlsx_bytes,
            )

        elapsed_seconds = time.monotonic() - request_started_at
        logger.info("job DONE job_id=%s company=%r mode=%s elapsed=%.2fs", job_id, company, mode, elapsed_seconds)
        db.log_job_run(company, mode, "done", f"elapsed={elapsed_seconds:.2f}s screening_id={screening_id}")

        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = "done"
            job["result"] = result
            job["files"] = files
            job["screening_id"] = screening_id
            job["created_at"] = time.monotonic()
        else:
            logger.warning(
                "job DONE but job_id no longer in JOBS job_id=%s company=%r screening_id=%s",
                job_id, company, screening_id,
            )

    except Exception as exc:
        logger.exception("job FAILED job_id=%s company=%r error=%s", job_id, company, exc)
        db.log_job_run(company, mode, "error", str(exc)[:2000])
        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = str(exc)
            job["created_at"] = time.monotonic()
    finally:
        if _INFLIGHT_JOB_BY_KEY.get(key) == job_id:
            _INFLIGHT_JOB_BY_KEY.pop(key, None)


async def _start_screen_job(company: str, mode: str, user_id: str | None) -> str:
    key = _job_key(company, mode)
    async with _JOBS_LOCK:
        existing_job_id = _INFLIGHT_JOB_BY_KEY.get(key)
        if existing_job_id and existing_job_id in JOBS and JOBS[existing_job_id]["status"] == "running":
            return existing_job_id

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "status": "running",
            "company": company,
            "mode": mode,
            "created_at": time.monotonic(),
            "result": None,
            "files": None,
            "error": None,
            "screening_id": None,
        }
        _INFLIGHT_JOB_BY_KEY[key] = job_id

    asyncio.create_task(_run_screen_job(job_id, company, mode, user_id))
    return job_id


@app.get("/")
async def index(request: Request, code: str = ""):
    if code:
        response, error = await _complete_oauth_login(code)
        if error:
            return templates.TemplateResponse(
                "login.html", {"request": request, "error": error}, status_code=401
            )
        return response

    user = _current_user(request)
    if auth.auth_configured() and not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/login")
async def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.get("/auth/google")
async def auth_google_start(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=302)
    oauth_url = auth.get_google_oauth_url(_redirect_uri(request))
    if not oauth_url:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Authentication is not configured."},
            status_code=500,
        )
    return RedirectResponse(oauth_url, status_code=302)


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = ""):
    if not code:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Google sign in did not complete. Please try again."},
            status_code=400,
        )
    response, error = await _complete_oauth_login(code)
    if error:
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": error}, status_code=401
        )
    return response


@app.post("/logout")
async def logout(request: Request):
    response = RedirectResponse("/login", status_code=302)
    auth.clear_session_cookie(response)
    return response


@app.get("/history")
async def history_page(request: Request):
    user = _current_user(request)
    if auth.auth_configured() and not user:
        return RedirectResponse("/login", status_code=302)
    rows = db.list_screenings(limit=100)
    return templates.TemplateResponse("history.html", {"request": request, "user": user, "screenings": rows})


@app.get("/api/history/search")
async def history_search(request: Request, q: str = ""):
    query = q.strip()
    if not query:
        return JSONResponse({"results": []})

    rows = db.search_screenings(query, limit=HISTORY_SEARCH_RESULT_LIMIT)
    return JSONResponse({"results": _rows_to_history_results(rows)})


@app.get("/results/{screening_id}")
async def results_page(request: Request, screening_id: str):
    row = db.get_screening(screening_id)
    if row is None:
        return HTMLResponse(
            '<div class="ff-error"><strong>That screening could not be found.</strong>'
            "<p>It may have been removed, or the link is out of date. Try searching again.</p></div>",
            status_code=404,
        )

    tier = {
        "label": row.get("tier_label"),
        "key": row.get("tier_key"),
        "color": row.get("tier_color"),
    }
    result = {
        "state": row.get("state"),
        "fit_score": row.get("fit_score"),
        "scoring_tier": tier,
        "analysis": row.get("analysis"),
        "score_breakdown": row.get("score_breakdown") or {},
        "decision_makers": row.get("decision_makers") or [],
        "sources": row.get("sources") or [],
        "source_links": [],
        "important_links": row.get("important_links") or [],
        "logo_url": row.get("logo_url"),
        "source_bank": row.get("source_bank") or [],
        "strategic_insight": row.get("strategic_insight", ""),
    }

    files = {}
    for file_row in db.get_screening_files(screening_id):
        signed_url = db.get_signed_file_url(file_row["storage_path"])
        files[file_row["label"]] = {
            "filename": file_row["filename"],
            "mime": file_row["mime_type"],
            "download_url": signed_url,
        }

    company = row.get("company", "")
    mode = row.get("mode", "screen")

    if _is_htmx_request(request):
        return render_results(request, company, mode, result, files)
    return render_results_page(request, company, mode, result, files)


@app.post("/screen")
async def screen(request: Request, company: str = Form(...), mode: str = Form("screen")):
    company = company.strip()
    if not company:
        return HTMLResponse(
            '<div class="ff-error"><strong>Enter a company name.</strong>'
            "<p>Type a company name before running research.</p></div>",
            status_code=400,
        )

    _prune_expired_jobs()

    user = _current_user(request)
    user_id = user["user_id"] if user else None

    reusable_row = _find_reusable_screening(company, mode)
    if reusable_row:
        logger.info(
            "screen reusing recent result company=%r mode=%s screening_id=%s",
            company, mode, reusable_row["id"],
        )
        return RedirectResponse(f"/results/{reusable_row['id']}", status_code=303)

    job_id = await _start_screen_job(company, mode, user_id)
    job = JOBS[job_id]

    return templates.TemplateResponse(
        "polling.html",
        {"request": request, "job_id": job_id, "company": job["company"], "mode": job["mode"]},
    )


@app.get("/screen/status/{job_id}")
async def screen_status(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return HTMLResponse(
            '<div class="ff-error"><strong>This research job could not be found.</strong>'
            "<p>It may have expired. Please run a new search.</p></div>",
            status_code=404,
        )

    if job["status"] == "running":
        return templates.TemplateResponse(
            "polling.html",
            {"request": request, "job_id": job_id, "company": job["company"], "mode": job["mode"], "still_working": True},
        )

    if job["status"] in ("error", "delivered_error"):
        error_message = job["error"]
        job["status"] = "delivered_error"
        job["created_at"] = time.monotonic()
        return HTMLResponse(
            '<div class="ff-error"><strong>Something went wrong running that request.</strong>'
            f'<p>{error_message}</p></div>'
        )

    screening_id = job.get("screening_id")
    job["status"] = "delivered"
    job["created_at"] = time.monotonic()

    if screening_id:
        return RedirectResponse(f"/results/{screening_id}", status_code=303)

    result = job["result"]
    files = job["files"]

    if _is_htmx_request(request):
        return render_results(request, job["company"], job["mode"], result, files)
    return render_results_page(request, job["company"], job["mode"], result, files)