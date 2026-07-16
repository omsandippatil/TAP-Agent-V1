from supabase import create_client, Client
from app.config import settings

_client: Client | None = None


def supabase_configured() -> bool:
    return bool(settings.supabase_url.strip() and settings.supabase_key.strip())


def get_client() -> Client | None:
    global _client
    if not supabase_configured():
        return None
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client