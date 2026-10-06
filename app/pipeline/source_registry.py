import re
import threading
from urllib.parse import urlparse

CORE_SOURCE_LABELS = {
    "india_csr_page": "Company CSR page",
    "mca_portal": "MCA portal",
    "mca_via_search": "MCA (via search)",
    "national_csr_portal": "National CSR Portal",
    "annual_report": "Annual / sustainability report",
    "global_annual_report": "Annual / sustainability report",
    "partner_search": "Funded partners search",
    "people_search": "LinkedIn people search",
    "plans_search": "Partnerships & plans search",
    "sector_eligibility_search": "Sector & eligibility search",
}


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


class SourceRegistry:
    def __init__(self, company: str):
        self.company = company
        self._entries: list[dict] = []
        self._url_to_number: dict[str, int] = {}
        self._name_to_number: dict[str, int] = {}
        self._lock = threading.Lock()

    def register(self, source_name: str, url: str = "", kind: str = "core",
                 label: str = "", excerpt: str = "", parent_number: int | None = None,
                 is_synthetic: bool = False) -> int:
        with self._lock:
            normalized_url = (url or "").strip()
            if normalized_url and normalized_url in self._url_to_number:
                return self._url_to_number[normalized_url]
            number = len(self._entries) + 1
            entry = {
                "number": number,
                "source_name": source_name,
                "kind": kind,
                "label": label or CORE_SOURCE_LABELS.get(source_name, source_name),
                "url": normalized_url,
                "domain": _domain_of(normalized_url),
                "excerpt": (excerpt or "").strip()[:280],
                "parent_number": parent_number,
                "is_synthetic": is_synthetic,
            }
            self._entries.append(entry)
            if normalized_url:
                self._url_to_number[normalized_url] = number
            if source_name and source_name not in self._name_to_number:
                self._name_to_number[source_name] = number
            return number

    def register_core_source(self, source: dict) -> int:
        if source.get("status") != "FOUND":
            return 0
        number = self.register(
            source_name=source.get("source_name", "source"),
            url=source.get("url", ""),
            kind="core",
            excerpt=source.get("text", ""),
            is_synthetic=bool(source.get("is_synthetic")),
        )
        source["source_number"] = number
        return number

    def register_child_hit(self, source_name: str, url: str, label: str, excerpt: str,
                            parent_number: int | None = None, is_synthetic: bool = False) -> int:
        return self.register(
            source_name=source_name,
            url=url,
            kind="child",
            label=label,
            excerpt=excerpt,
            parent_number=parent_number,
            is_synthetic=is_synthetic,
        )

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def as_manifest_lines(self) -> list[str]:
        lines = []
        for e in self.entries():
            synthetic_tag = " (synthesized fact, not a direct fetch)" if e.get("is_synthetic") else ""
            lines.append(f"[{e['number']}] {e['label']}{synthetic_tag} — {e['domain'] or e['url']}")
        return lines

    def as_source_bank(self) -> list[dict]:
        return [
            {
                "number": e["number"], "label": e["label"], "source_name": e["source_name"],
                "kind": e["kind"], "url": e["url"], "domain": e["domain"],
                "excerpt": e["excerpt"], "parent_number": e["parent_number"],
                "is_synthetic": e.get("is_synthetic", False),
            }
            for e in self.entries()
        ]


CITATION_TOKEN_PATTERN = re.compile(r"\[(\d{1,3})\]")


def strip_unknown_citation_tokens(text: str, valid_numbers: set[int]) -> str:
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        number = int(match.group(1))
        return match.group(0) if number in valid_numbers else ""

    return CITATION_TOKEN_PATTERN.sub(_replace, text)


def extract_cited_numbers(text: str) -> list[int]:
    if not text:
        return []
    seen = []
    for match in CITATION_TOKEN_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen