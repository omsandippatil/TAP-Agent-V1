from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SESSION_SECRET = "change-me-in-prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    supabase_key: str = ""
    supabase_anon_key: str = ""
    app_env: str = "development"
    session_secret: str = DEFAULT_SESSION_SECRET
    session_cookie_name: str = "ff_session"
    session_max_age_seconds: int = 60 * 60 * 24 * 30
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_context_window: int = 200000
    google_search_api_key: str = ""
    google_search_engine_id: str = ""
    google_search_daily_cap: int = 90
    config_yaml_path: str = "config.yaml"
    verbose_pipeline_logging: bool = False
    llm_dump_dir: str = "/tmp/fundfinder_llm_dumps"
    enable_directed_search: bool = True
    max_directed_search_queries_screen: int = 4
    max_directed_search_queries_deep: int = 8
    directed_search_medium_priority_enabled: bool = True
    screen_runs_per_hour: int = 5

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    @property
    def google_search_api_keys(self) -> list[str]:
        return [k.strip() for k in self.google_search_api_key.split(",") if k.strip()]

    @property
    def google_search_configured(self) -> bool:
        return bool(self.google_search_api_keys and self.google_search_engine_id.strip())

    @property
    def supabase_auth_configured(self) -> bool:
        return bool(self.supabase_url.strip() and self.supabase_anon_key.strip())


settings = Settings()

if settings.app_env == "production" and settings.session_secret == DEFAULT_SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET must be set when APP_ENV=production")