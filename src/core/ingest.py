from dataclasses import asdict, dataclass
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from utils.logger import get_logger


BASE_URL = "https://paulgraham.com/"
INDEX_URL = urljoin(BASE_URL, "articles.html")

CATALOG_PATH = Path("data/catalog/catalog.json")
ESSAY_DIR = Path("data/essays")
ingest_logger = get_logger("ingest_core", folder="pipeline")


@dataclass
class EssayLink:
    index: int
    title: str
    url: str
    slug: str


@dataclass
class EssayDocument:
    title: str
    url: str
    slug: str
    text: str


def fetch_essay_list() -> list[EssayLink]:
    ingest_logger.info("Fetching essay index from %s", INDEX_URL)
    response = requests.get(INDEX_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    essays = []
    seen = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href")
        if not isinstance(href, str):
            ingest_logger.warning("Skipping anchor with non-string href")
            continue
        href = href.strip()
        title = a_tag.get_text(strip=True)

        if not href or not title:
            continue

        url = urljoin(BASE_URL, href)
        parsed = urlparse(url)
        filename = Path(parsed.path).name

        if parsed.netloc and parsed.netloc != "paulgraham.com":
            continue

        if not filename.endswith(".html") or filename == "articles.html":
            continue

        slug = Path(filename).stem

        if url in seen:
            continue

        essays.append(EssayLink(index=len(essays), title=title, url=url, slug=slug))
        seen.add(url)

    ingest_logger.info("Discovered %s essays", len(essays))
    return essays


def write_list(essays: list[EssayLink], path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(essay) for essay in essays]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ingest_logger.info("Wrote essay catalog to %s", path)


def read_list(path: Path = CATALOG_PATH) -> list[EssayLink]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ingest_logger.info("Loaded %s essays from %s", len(payload), path)
    return [EssayLink(**item) for item in payload]


def select_essays(
    essays: list[EssayLink],
    slug: str | None = None,
    position: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> list[EssayLink]:
    if slug is not None:
        selected = [essay for essay in essays if essay.slug == slug]
        if not selected:
            ingest_logger.warning("No essay found for slug=%s", slug)
        return selected

    if position is not None:
        if position >= len(essays):
            ingest_logger.error("Position %s out of range for %s essays", position, len(essays))
            raise IndexError("Essay position out of range.")
        selected = [essay for essay in essays if essay.index == position]
        if not selected:
            ingest_logger.error("No essay found for position=%s", position)
            raise IndexError("Essay position out of range.")
        return selected

    if start is None and end is None:
        return essays

    start = 0 if start is None else start
    end = start + 1 if end is None else end

    selected = essays[start:end]
    if not selected:
        ingest_logger.warning("Essay selection returned no results for start=%s end=%s", start, end)
    return selected


def fetch_essay(link: EssayLink) -> EssayDocument:
    ingest_logger.info("Fetching essay %s from %s", link.slug, link.url)
    response = requests.get(link.url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else link.title
    body = soup.body.get_text(separator="\n") if soup.body else ""
    text = "\n\n".join(line.strip() for line in body.splitlines() if line.strip())
    if not text:
        ingest_logger.error("No text extracted for essay %s", link.slug)
        raise ValueError(f"No text extracted from {link.url}")

    return EssayDocument(
        title=title,
        url=link.url,
        slug=link.slug,
        text=text,
    )


def write_essay(doc: EssayDocument, out_dir: Path = ESSAY_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.slug}.json"
    path.write_text(json.dumps(asdict(doc), indent=2) + "\n", encoding="utf-8")
    ingest_logger.info("Wrote essay document to %s", path)


def discover_catalog() -> Path:
    essays = fetch_essay_list()
    write_list(essays)
    return CATALOG_PATH


def ingest_documents(
    slug: str | None = None,
    position: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> list[EssayDocument]:
    if not CATALOG_PATH.exists():
        ingest_logger.warning("Catalog file missing, running discovery first")
        discover_catalog()

    essays = read_list()
    selected = select_essays(
        essays=essays,
        slug=slug,
        position=position,
        start=start,
        end=end,
    )

    documents: list[EssayDocument] = []
    for essay in selected:
        document = fetch_essay(essay)
        write_essay(document)
        documents.append(document)
        ingest_logger.info("Completed ingest for %s", document.slug)

    ingest_logger.info("Ingested %s documents", len(documents))

    return documents


if __name__ == "__main__":
    discover_catalog()
    ingest_documents(position=0)
