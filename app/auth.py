import logging
import time

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from supabase import create_client, Client

from fastapi import Request

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
    except SignatureExpired:
        return None
    except BadSignature:
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
    return _session_payload_from_response(response), None


def _session_payload_from_response(response) -> dict:
    user = response.user
    session = response.session
    return {
        "user_id": user.id,
        "email": user.email,
        "access_token": session.access_token if session else None,
        "refresh_token": session.refresh_token if session else None,
        "issued_at": int(time.time()),
    }


def refresh_session(refresh_token: str) -> dict | None:
    client = get_anon_client()
    if client is None or not refresh_token:
        return None
    try:
        response = client.auth.refresh_session(refresh_token)
    except Exception as exc:
        logger.info("refresh_session failed error=%s", exc)
        return None
    if response.user is None or response.session is None:
        return None
    return _session_payload_from_response(response)


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    session = decode_session(token)
    if session is None:
        return None
    return session


def set_session_cookie(response, session_payload: dict) -> None:
    token = encode_session(session_payload)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.app_env == "production",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(settings.session_cookie_name)