from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    model_version: str = "v1.0"
    host: str = "0.0.0.0"
    port: int = 5002
    log_level: str = "INFO"

settings = Settings()