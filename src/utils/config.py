from dataclasses import dataclass
import os
from dotenv import load_dotenv
from pathlib import Path

from utils.logger import get_logger

load_dotenv()
config_logger = get_logger("config", folder="system")


@dataclass(frozen=True)
class Settings:
    """Minimal settings needed to grow the project safely."""

    project_root: Path
    catalog_dir: Path
    catalog_path: Path
    selected_essays_path: Path
    raw_data_dir: Path
    processed_data_dir: Path
    chunks_path: Path
    chroma_dir: Path
    chroma_collection_name: str
    results_dir: Path
    paul_graham_index_url: str
    openai_api_key: str
    gemini_api_key: str
    gemini_judge_model: str


def load_settings() -> Settings:
    """Load runtime settings from the environment."""

    project_root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=project_root,
        catalog_dir=project_root / "data" / "catalog",
        catalog_path=project_root / "data" / "catalog" / "paul_graham_index.json",
        selected_essays_path=project_root / "data" / "catalog" / "selected_essays.json",
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        chunks_path=project_root / "data" / "chunks" / "recursive_chunks.json",
        chroma_dir=project_root / "data" / "chroma",
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "essay_chunks"),
        results_dir=project_root / "results",
        paul_graham_index_url=os.getenv(
            "PAUL_GRAHAM_INDEX_URL",
            "https://paulgraham.com/articles.html",
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_judge_model=os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash"),
    )

    config_logger.info("Settings loaded from %s", project_root)
    if not settings.openai_api_key:
        config_logger.warning("OPENAI_API_KEY is not set")
    if not settings.gemini_api_key:
        config_logger.warning("GEMINI_API_KEY is not set")
    if not Path(".env").exists():
        config_logger.warning(".env file not found in project root")

    return settings
