from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    database_url: str = ""
    app_env: str = "development"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    anthropic_tpm_limit: int = 40000

    google_search_api_key: str = ""
    google_search_engine_id: str = ""
    google_search_daily_cap: int = 90

    config_yaml_path: str = "config.yaml"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    @property
    def google_search_configured(self) -> bool:
        return bool(self.google_search_api_key.strip() and self.google_search_engine_id.strip())


settings = Settings()