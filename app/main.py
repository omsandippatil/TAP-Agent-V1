import asyncio
import base64
import logging
import time
import uuid

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import scorer, scraper
from app.pipeline.config_loader import load_config
from app.render.xlsx_reporter import generate_deep_dive_xlsx
from app.render.docx_reporter import generate_docx_report
from app.render.reporters import render_results, templates

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


def _encode_file(content_bytes: bytes, filename: str, mime: str) -> tuple[str, str, str]:
    b64 = base64.b64encode(content_bytes).decode("ascii")
    return filename, mime, b64


def _slugify(company: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in company.strip().lower()).strip("_") or "company"


def _prune_expired_jobs():
    now = time.monotonic()
    expired = [job_id for job_id, job in JOBS.items() if now - job["created_at"] > JOB_TTL_SECONDS]
    for job_id in expired:
        JOBS.pop(job_id, None)


async def _run_screen_job(job_id: str, company: str, mode: str):
    job = JOBS[job_id]
    try:
        request_started_at = time.monotonic()
        logger.info("job START job_id=%s company=%r mode=%s", job_id, company, mode)

        cfg = load_config()
        search_cfg = cfg.get("search_source_toggles", {})

        if mode == "deep":
            sources = await scraper.fetch_deep_sources(company, search_cfg)
        else:
            sources = await scraper.fetch_screen_sources(company, search_cfg)

        found_count = sum(1 for s in sources if s.get("status") == "FOUND")
        logger.info("job sources job_id=%s company=%r found=%d/%d", job_id, company, found_count, len(sources))

        result = await scorer.score(company, sources, cfg)
        logger.info(
            "job scored job_id=%s company=%r state=%s fit_score=%s analysis_present=%s",
            job_id, company, result.get("state"), result.get("fit_score"), bool(result.get("analysis")),
        )

        files = {}
        if mode == "deep":
            slug = _slugify(company)
            docx_bytes = await generate_docx_report(company, result, mode)
            files["Word Report (.docx)"] = _encode_file(
                docx_bytes,
                f"tap_csr_{slug}_deep.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            xlsx_bytes = await generate_deep_dive_xlsx(company, result, cfg)
            files["Deep-Dive Workbook (.xlsx)"] = _encode_file(
                xlsx_bytes,
                f"tap_csr_{slug}_deep_dive.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        elapsed_seconds = time.monotonic() - request_started_at
        logger.info("job DONE job_id=%s company=%r mode=%s elapsed=%.2fs", job_id, company, mode, elapsed_seconds)

        job["status"] = "done"
        job["result"] = result
        job["files"] = files
    except Exception as exc:
        logger.exception("job FAILED job_id=%s company=%r error=%s", job_id, company, exc)
        job["status"] = "error"
        job["error"] = str(exc)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/screen")
async def screen(request: Request, company: str = Form(...), mode: str = Form("screen")):
    company = company.strip()
    _prune_expired_jobs()

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "running",
        "company": company,
        "mode": mode,
        "created_at": time.monotonic(),
        "result": None,
        "files": None,
        "error": None,
    }

    asyncio.create_task(_run_screen_job(job_id, company, mode))

    return templates.TemplateResponse(
        "polling.html",
        {"request": request, "job_id": job_id, "company": company, "mode": mode},
    )


@app.get("/screen/{job_id}")
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

    if job["status"] == "error":
        return HTMLResponse(
            '<div class="ff-error"><strong>Something went wrong running that request.</strong>'
            f'<p>{job["error"]}</p></div>'
        )

    result = job["result"]
    files = job["files"]
    company = job["company"]
    mode = job["mode"]
    JOBS.pop(job_id, None)
    return render_results(request, company, mode, result, files)