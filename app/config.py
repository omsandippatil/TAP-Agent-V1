from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    database_url: str = ""
    app_env: str = "development"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile" 
    groq_tpm_limit: int = 12000

    google_search_api_key: str = ""
    google_search_engine_id: str = ""
    google_search_daily_cap: int = 90

    config_yaml_path: str = "config.yaml"

    class Config:
        env_file = ".env"

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def google_search_configured(self) -> bool:
        return bool(self.google_search_api_key.strip() and self.google_search_engine_id.strip())


settings = Settings()