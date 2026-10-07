"""
Authentication Routes

Login/logout/change-password endpoints + HTML pages.
Matched with the finalized Peewee SQL Backend and stdout Audit Logging.
"""

import hashlib
import hmac
import logging

from itsdangerous import URLSafeTimedSerializer
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route, Router

from dagster_authkit.api.health import track_login_attempt, track_session_created
from dagster_authkit.auth.rate_limiter import (
    get_rate_limiter,
    reset_rate_limit,
)
from dagster_authkit.auth.security import SecurityHardening
from dagster_authkit.auth.session import sessions
from dagster_authkit.core.registry import get_backend
from dagster_authkit.utils.audit import (
    log_audit_event,
    log_login_attempt,
    log_logout,
    log_rate_limit_violation,
)
from dagster_authkit.utils.config import config
from dagster_authkit.utils.templates import render_change_password_page, render_login_page

logger = logging.getLogger(__name__)

# Stateless synchronizer token pattern (valid for 1 hour).
# Tokens are signed, not stored server-side — works across K8s pods.
_csrf_serializer = URLSafeTimedSerializer(config.SECRET_KEY)
_CSRF_MAX_AGE = 3600

# Backends whose users may change their own password from the UI.
PASSWORD_CHANGE_BACKENDS = ("sql", "sqlite")


def _generate_csrf_token() -> str:
    """Generate a signed CSRF token valid for 1 hour."""
    raw = SecurityHardening.generate_csrf_token()
    return _csrf_serializer.dumps({"token": raw})


def _validate_csrf_token(token: str, cookie: str = "") -> bool:
    """Validate a signed CSRF token via double-submit cookie pattern.

    Checks that:
    1. The form token matches the cookie value (double-submit binding).
    2. The token is a valid signed blob and not expired.

    Args:
        token: CSRF token from the form submission.
        cookie: CSRF token from the cookie (double-submit check).

    Returns:
        True if both checks pass.
    """
    if not token:
        return False
    if cookie and not hmac.compare_digest(token, cookie):
        return False
    try:
        _csrf_serializer.loads(token, max_age=_CSRF_MAX_AGE)
        return True
    except Exception:
        return False


def _get_client_ip(request: Request) -> str:
    """Get real client IP, only trusting X-Forwarded-For from known proxies."""
    client_host = request.client.host if request.client else "unknown"

    # Only trust X-Forwarded-For if the connection comes from a trusted proxy.
    # Otherwise, an attacker can spoof the header to bypass IP-based rate limiting.
    if client_host in config.DAGSTER_AUTH_PROXY_TRUSTED_IPS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return client_host


async def login_page(request: Request) -> Response:
    """GET /auth/login - Displays login page."""
    next_url = request.query_params.get("next", "/")
    if not SecurityHardening.validate_redirect_url(next_url):
        next_url = "/"

    error = request.query_params.get("error", "")

    # Reuse an existing, still-valid CSRF cookie rather than always minting a
    # new one. Without this, any incidental request to a protected path while
    # the login page is open (like a browser's automatic /favicon.ico call)
    # silently rotates the cookie and invalidates the token
    # already embedded in the visible login form. The user then gets
    # "CSRF validation failed" on submit despite doing nothing wrong.
    existing_cookie = request.cookies.get("csrf_token")
    if existing_cookie is not None and _validate_csrf_token(existing_cookie):
        csrf_token = existing_cookie
        needs_cookie = False
    else:
        csrf_token = _generate_csrf_token()
        needs_cookie = True

    html = render_login_page(next_url, error, csrf_token)

    response = HTMLResponse(content=html)
    if needs_cookie:
        response.set_cookie(
            key="csrf_token",
            value=csrf_token,
            max_age=_CSRF_MAX_AGE,
            httponly=True,
            secure=config.SESSION_COOKIE_SECURE,
            samesite="lax",
        )
    return response


async def process_login(request: Request) -> Response:
    """
    POST /auth/process — Authenticates and creates session.

    Multi-step flow:
    1. Validate CSRF token (stateless signed, 1h TTL).
    2. Check rate limits for both username and IP independently.
    3. Authenticate via the configured backend (runs in thread pool).
    4. On success: create session, set cookie, redirect to next_url.
    5. On failure: record attempt, redirect with error.

    Args:
        request: Starlette Request with form fields (username, password,
                 next, csrf_token).

    Returns:
        RedirectResponse (302) to either next_url on success or
        /auth/login with error message on failure.
    """
    form = await request.form()
    username = SecurityHardening.sanitize_username(str(form.get("username", "")).strip())
    password = str(form.get("password", ""))
    next_url = str(form.get("next", "/"))

    if not SecurityHardening.validate_redirect_url(next_url):
        next_url = "/"

    # CSRF validation: double-submit cookie + signed token
    csrf_token = str(form.get("csrf_token", ""))
    csrf_cookie = request.cookies.get("csrf_token", "")
    if not _validate_csrf_token(csrf_token, cookie=csrf_cookie):
        logger.warning("CSRF validation failed for login attempt", exc_info=True)
        track_login_attempt(False, username)
        return RedirectResponse(
            url=f"/auth/login?next={next_url}&error=Invalid+request.", status_code=302
        )

    client_ip = _get_client_ip(request)
    ip_identifier = f"ip:{client_ip}"
    rl = get_rate_limiter()

    # 1. Rate Limiting Check (username + IP, independent, atomic)
    user_limited, user_attempts = rl.check_and_record(username)
    ip_limited, ip_attempts = rl.check_and_record(ip_identifier)
    if user_limited or ip_limited:
        total = max(user_attempts, ip_attempts)
        log_login_attempt(username, False, client_ip, f"RATE_LIMIT ({total} attempts)")
        log_rate_limit_violation(username, client_ip, total)
        track_login_attempt(False, username)
        return RedirectResponse(
            url=f"/auth/login?next={next_url}&error=Too+many+attempts.", status_code=302
        )

    # 2. Backend Call (Peewee / SQL / LDAP) — runs in thread pool to avoid
    #    blocking the async event loop (bcrypt ~250ms + DB/LDAP I/O).
    try:
        backend = get_backend(config.AUTH_BACKEND, config.__dict__)
        user = await run_in_threadpool(backend.authenticate, username, password)
    except Exception as e:
        logger.exception(f"Auth Backend Error: {e}")
        log_login_attempt(username, False, client_ip, "BACKEND_ERROR")
        track_login_attempt(False, username)
        return RedirectResponse(
            url=f"/auth/login?next={next_url}&error=System+error.", status_code=302
        )

    # 3. Validation
    if not user:
        # Attempt already recorded by check_and_record above
        log_login_attempt(username, False, client_ip, "INVALID_CREDENTIALS")
        track_login_attempt(False, username)
        return RedirectResponse(
            url=f"/auth/login?next={next_url}&error=Invalid+credentials.", status_code=302
        )

    # 4. Success & Session Creation
    reset_rate_limit(username)
    reset_rate_limit(ip_identifier)
    log_login_attempt(username, True, client_ip)
    track_login_attempt(True, username)

    session_token = sessions.create(user.to_dict())
    track_session_created()

    session_hash = hashlib.sha256(session_token.encode()).hexdigest()[:16]
    log_audit_event("SESSION_CREATED", username, session_id=session_hash, ip=client_ip)

    response = RedirectResponse(url=next_url, status_code=302)
    response.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=session_token,
        max_age=config.SESSION_MAX_AGE,
        httponly=config.SESSION_COOKIE_HTTPONLY,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
    )
    return response


async def logout(request: Request) -> Response:
    """
    GET /auth/logout - Revokes session.

    Behavior:
    - Proxy mode: Redirects to Authelia logout URL
    - Session mode: Revokes local session cookie
    """
    if config.AUTH_BACKEND == "proxy":
        logout_url = config.DAGSTER_AUTH_PROXY_LOGOUT_URL
        logger.info(f"Proxy mode: Redirecting logout to {logout_url}")
        return RedirectResponse(url=logout_url, status_code=302)

    session_token = request.cookies.get(config.SESSION_COOKIE_NAME)
    username = "unknown"

    if session_token:
        user_data = sessions.validate(session_token)
        if user_data:
            username = user_data.get("username", "unknown")
            sessions.revoke(session_token)

    client_ip = _get_client_ip(request)
    log_logout(username, client_ip)

    response = RedirectResponse(url="/auth/login", status_code=302)
    response.delete_cookie(
        key=config.SESSION_COOKIE_NAME,
        httponly=config.SESSION_COOKIE_HTTPONLY,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
    )
    response.delete_cookie(
        key="csrf_token",
        httponly=True,
        secure=config.SESSION_COOKIE_SECURE,
        samesite="lax",
    )
    return response


def _get_session_user(request: Request) -> dict:
    """Return the validated session payload of the current request, or {} if none."""
    session_token = request.cookies.get(config.SESSION_COOKIE_NAME)
    if not session_token:
        return {}
    return sessions.validate(session_token) or {}


def _change_password_response(
    request: Request,
    username: str,
    error: str = "",
    success: bool = False,
    status_code: int = 200,
) -> Response:
    """Render the change password page, issuing a CSRF cookie when needed."""
    # Same reuse-or-mint logic as login_page: do not rotate a valid cookie.
    existing_cookie = request.cookies.get("csrf_token")
    if existing_cookie is not None and _validate_csrf_token(existing_cookie):
        csrf_token = existing_cookie
        needs_cookie = False
    else:
        csrf_token = _generate_csrf_token()
        needs_cookie = True

    html = render_change_password_page(
        username,
        csrf_token=csrf_token,
        error=error,
        success=success,
        min_length=config.PASSWORD_MIN_LENGTH,
    )
    response = HTMLResponse(content=html, status_code=status_code)
    # The page holds credentials-related state; never serve it from cache.
    response.headers["Cache-Control"] = "no-store"
    if needs_cookie:
        response.set_cookie(
            key="csrf_token",
            value=csrf_token,
            max_age=_CSRF_MAX_AGE,
            httponly=True,
            secure=config.SESSION_COOKIE_SECURE,
            samesite="lax",
        )
    return response


_CHANGE_PASSWORD_LOGIN_REDIRECT = "/auth/login?next=/auth/change-password"


async def change_password_page(request: Request) -> Response:
    """GET /auth/change-password - Displays the change password form."""
    if config.AUTH_BACKEND not in PASSWORD_CHANGE_BACKENDS:
        return Response(content="Not Found", status_code=404)

    # /auth/ is a public prefix in the middleware, so the session is checked here.
    username = _get_session_user(request).get("username")
    if not username:
        return RedirectResponse(url=_CHANGE_PASSWORD_LOGIN_REDIRECT, status_code=302)

    success = request.query_params.get("success") == "1"
    return _change_password_response(request, username, success=success)


async def process_change_password(request: Request) -> Response:
    """
    POST /auth/change-password — Changes the password of the signed-in user.

    The target account is always the session's user; the form only carries
    the current password and the new password twice. Nothing but the
    password (and the session version, to sign out other devices) changes.

    Flow:
    1. Require a supported backend and a valid session.
    2. Validate CSRF token (double-submit cookie, cookie mandatory).
    3. Validate the form (confirmation matches, password policy).
    4. Rate limit, then verify the current password via the backend.
    5. Change the password, rotate the session, redirect back with success.

    Returns:
        303 redirect to the form on success, the form with an error
        (4xx) otherwise, or a 302 redirect to login without a session.
    """
    if config.AUTH_BACKEND not in PASSWORD_CHANGE_BACKENDS:
        return Response(content="Not Found", status_code=404)

    user_data = _get_session_user(request)
    username = user_data.get("username")
    if not username:
        return RedirectResponse(url=_CHANGE_PASSWORD_LOGIN_REDIRECT, status_code=302)

    client_ip = _get_client_ip(request)

    def fail(reason: str, message: str, status_code: int = 400) -> Response:
        log_audit_event("PASSWORD_CHANGE_FAILED", username, reason=reason, ip=client_ip)
        return _change_password_response(request, username, error=message, status_code=status_code)

    form = await request.form()

    # CSRF validation: unlike login, the cookie is mandatory here because the
    # request is authenticated and state-changing.
    csrf_token = str(form.get("csrf_token", ""))
    csrf_cookie = request.cookies.get("csrf_token", "")
    if not csrf_cookie or not _validate_csrf_token(csrf_token, cookie=csrf_cookie):
        logger.warning("CSRF validation failed for password change")
        return fail("CSRF", "Invalid request. Please try again.", 403)

    current_password = str(form.get("current_password", ""))
    new_password = str(form.get("new_password", ""))
    new_password_confirm = str(form.get("new_password_confirm", ""))

    if not current_password or not new_password:
        return fail("MISSING_FIELDS", "All fields are required.")

    if new_password != new_password_confirm:
        return fail("CONFIRMATION_MISMATCH", "New passwords do not match.")

    policy_error = SecurityHardening.validate_password_strength(
        new_password, config.PASSWORD_MIN_LENGTH
    )
    if policy_error:
        return fail("WEAK_PASSWORD", policy_error)

    if new_password == current_password:
        return fail("SAME_PASSWORD", "New password must differ from the current one.")

    # Rate limit guesses of the current password (e.g. from a hijacked session).
    # Separate key so it does not interfere with the login limiter.
    rl_identifier = f"pwchange:{username}"
    rl = get_rate_limiter()
    limited, attempts = rl.check_and_record(rl_identifier)
    if limited:
        log_rate_limit_violation(username, client_ip, attempts)
        return fail("RATE_LIMIT", "Too many attempts. Please try again later.", 429)

    try:
        backend = get_backend(config.AUTH_BACKEND, config.__dict__)
        if not await run_in_threadpool(backend.verify_password, username, current_password):
            return fail("INVALID_CURRENT_PASSWORD", "Current password is incorrect.")

        changed = await run_in_threadpool(
            backend.change_password, username, new_password, performed_by=username
        )
    except Exception as e:
        logger.exception(f"Auth Backend Error during password change: {e}")
        return fail("BACKEND_ERROR", "System error. Please try again later.", 500)

    if not changed:
        return fail("BACKEND_ERROR", "System error. Please try again later.", 500)

    rl.reset(rl_identifier)

    # change_password invalidated every session of the user (other devices are
    # signed out); keep this browser signed in with a fresh session.
    session_data = {k: v for k, v in user_data.items() if k != "_v"}
    try:
        session_token = sessions.rotate(session_data)
    except Exception as e:
        # The password is already changed; only the new session could not be
        # issued (e.g. session store outage). Send the user to sign in again
        # instead of failing the request.
        logger.exception(f"Session rotation failed after password change: {e}")
        log_audit_event("SESSION_ROTATION_FAILED", username, ip=client_ip)
        response = RedirectResponse(
            url="/auth/login?next=/&error=Password+changed.+Please+sign+in+again.",
            status_code=303,
        )
        response.delete_cookie(
            key=config.SESSION_COOKIE_NAME,
            httponly=config.SESSION_COOKIE_HTTPONLY,
            secure=config.SESSION_COOKIE_SECURE,
            samesite=config.SESSION_COOKIE_SAMESITE,
        )
        return response
    track_session_created()

    session_hash = hashlib.sha256(session_token.encode()).hexdigest()[:16]
    log_audit_event("SESSION_CREATED", username, session_id=session_hash, ip=client_ip)

    response = RedirectResponse(url="/auth/change-password?success=1", status_code=303)
    response.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=session_token,
        max_age=config.SESSION_MAX_AGE,
        httponly=config.SESSION_COOKIE_HTTPONLY,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
    )
    return response


def create_auth_routes() -> Router:
    """
    Creates router with authentication routes.

    Returns:
        Starlette Router with routes /login, /logout, /process, /change-password
    """
    return Router(
        routes=[
            Route("/login", login_page, methods=["GET"]),
            Route("/process", process_login, methods=["POST"]),
            Route("/logout", logout, methods=["GET"]),
            Route("/change-password", change_password_page, methods=["GET"]),
            Route("/change-password", process_change_password, methods=["POST"]),
        ]
    )
