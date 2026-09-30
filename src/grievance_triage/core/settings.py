from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    neon_db: str
    google_api_key: str
    model_checkpoint: str = "outputs/submission_muril_model.pt"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    complaint_similarity_threshold: float = 0.70
    similarity_top_k: int = 5
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    database_pool_size: int = 5
    database_max_overflow: int = 10

    @property
    def database_url(self) -> str:
        url = self.neon_db
        return url.replace("postgresql://", "postgresql+psycopg://", 1)

    @property
    def checkpoint_path(self) -> Path:
        # Prefer fp16 compressed model (452MB) over original (905MB)
        fp16_path = PROJECT_ROOT / "outputs" / "submission_muril_model_fp16.pt"
        if fp16_path.exists():
            return fp16_path
        muril_path = PROJECT_ROOT / "outputs" / "submission_muril_model.pt"
        if muril_path.exists():
            return muril_path
        path = Path(self.model_checkpoint)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
