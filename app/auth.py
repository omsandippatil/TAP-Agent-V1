import logging
import time

from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from supabase import Client, create_client

from app.config import settings

logger = logging.getLogger("tap.auth")

_anon_client: Client | None = None


def auth_configured() -> bool:
    return settings.supabase_auth_configured


def get_anon_client() -> Client | None:
    global _anon_client
    if not auth_configured():
        return None
    if _anon_client is None:
        _anon_client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _anon_client


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt="ff-session")


def encode_session(payload: dict) -> str:
    return _serializer().dumps(payload)


def decode_session(token: str) -> dict | None:
    try:
        return _serializer().loads(token, max_age=settings.session_max_age_seconds)
    except (SignatureExpired, BadSignature):
        return None


def get_google_oauth_url(redirect_to: str) -> str | None:
    client = get_anon_client()
    if client is None:
        return None
    response = client.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": redirect_to},
    })
    return response.url


def exchange_code_for_session(auth_code: str) -> tuple[dict | None, str | None]:
    client = get_anon_client()
    if client is None:
        return None, "Authentication is not configured."
    try:
        response = client.auth.exchange_code_for_session({"auth_code": auth_code})
    except Exception as exc:
        logger.warning("exchange_code_for_session failed error=%s", exc)
        return None, "Could not complete sign in with Google."
    if response.user is None or response.session is None:
        return None, "Could not complete sign in with Google."
    return {
        "user_id": response.user.id,
        "email": response.user.email,
        "issued_at": int(time.time()),
    }, None


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(settings.session_cookie_name)
    return decode_session(token) if token else None


def set_session_cookie(response, session_payload: dict) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=encode_session(session_payload),
        max_age=settings.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.app_env == "production",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(settings.session_cookie_name)