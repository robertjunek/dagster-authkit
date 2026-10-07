"""
End-to-end tests for the self-service password change (/auth/change-password).

The real auth router is mounted behind the real DagsterAuthMiddleware and
talks to a real PeeweeAuthBackend and a real session backend, driven through
Starlette's TestClient. Only the backend registry lookup and the rate limiter
singleton are swapped for per-test instances.

The whole suite runs once per storage stack (database x session store).
SQLite with cookie sessions always runs; the other stacks need live services
and are skipped unless their URL is provided:

    AUTHKIT_TEST_POSTGRES_URL=postgresql://user:pass@host:5432/db
    AUTHKIT_TEST_MYSQL_URL=mysqlconnector://user:pass@host:3306/db
    AUTHKIT_TEST_REDIS_URL=redis://host:6379/15

Point these at disposable instances only: the ``users`` table is dropped and
the Redis database is flushed around every test.

Covers:
- Access control: session required, only the signed-in user's account
- Happy path: password changed, current browser stays signed in,
  other sessions revoked, nothing else in the user record changes
- Validation: current password, confirmation, policy, same password
- CSRF (double-submit cookie, cookie mandatory)
- Rate limiting of current-password guesses
- Unsupported backends, backend failures
- Audit trail and absence of plaintext passwords in logs
"""

import json
import logging
import os
import re

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from dagster_authkit.api.routes import _generate_csrf_token, create_auth_routes
from dagster_authkit.auth.backends.base import AuthUser, Role
from dagster_authkit.auth.backends.sql import PeeweeAuthBackend, UserTable
from dagster_authkit.auth.rate_limiter import RateLimiter, get_rate_limiter
from dagster_authkit.auth.session import CookieBackend, RedisBackend, sessions
from dagster_authkit.core.middleware import DagsterAuthMiddleware
from dagster_authkit.utils.audit import audit_logger
from dagster_authkit.utils.config import config

URL = "/auth/change-password"
OLD = "OldPassw0rd"
NEW = "NewPassw0rd"
BOB_PASSWORD = "BobPassw0rd"
# Rate limiter threshold. As for login, the attempt that reaches the maximum
# is already refused, so MAX_ATTEMPTS - 1 guesses are actually evaluated.
MAX_ATTEMPTS = 4
ALLOWED_GUESSES = MAX_ATTEMPTS - 1

DATABASE_URL_ENV = {
    "postgresql": "AUTHKIT_TEST_POSTGRES_URL",
    "mysql": "AUTHKIT_TEST_MYSQL_URL",
}
REDIS_URL_ENV = "AUTHKIT_TEST_REDIS_URL"

STACKS = [
    pytest.param((database, session_store), id=f"{database}-{session_store}")
    for database in ("sqlite", "postgresql", "mysql")
    for session_store in ("cookie", "redis")
]


@pytest.fixture(autouse=True)
def fast_bcrypt(monkeypatch):
    """Lower the bcrypt cost: these tests hash hundreds of passwords."""
    try:
        import bcrypt
    except ImportError:
        return
    gensalt = bcrypt.gensalt
    monkeypatch.setattr(bcrypt, "gensalt", lambda rounds=12, prefix=b"2b": gensalt(4, prefix))


@pytest.fixture(params=STACKS)
def stack(request):
    """(database, session store) pair; skips when its service is not configured."""
    database, session_store = request.param
    if database != "sqlite" and not os.getenv(DATABASE_URL_ENV[database]):
        pytest.skip(f"{DATABASE_URL_ENV[database]} not set")
    if session_store == "redis" and not os.getenv(REDIS_URL_ENV):
        pytest.skip(f"{REDIS_URL_ENV} not set")
    return database, session_store


@pytest.fixture
def backend(stack, tmp_path, monkeypatch):
    """Real SQL backend wired in as the configured backend, on the stack's storage."""
    database, session_store = stack

    if database == "sqlite":
        # A file (not :memory:) is required: the routes hit the database from
        # worker threads and every thread gets its own SQLite connection.
        database_url = f"sqlite:///{tmp_path / 'auth.db'}"
    else:
        database_url = os.environ[DATABASE_URL_ENV[database]]

    be = PeeweeAuthBackend({"DAGSTER_AUTH_DATABASE_URL": database_url})
    # Start from a clean table even if a previous run was interrupted.
    UserTable.drop_table(safe=True)
    be.db.create_tables([UserTable])
    be.add_user("alice", OLD, Role.VIEWER, email="alice@example.com", full_name="Alice A")
    be.add_user("bob", BOB_PASSWORD, Role.ADMIN, email="bob@example.com", full_name="Bob B")

    monkeypatch.setattr(config, "AUTH_BACKEND", "sql")
    monkeypatch.setattr("dagster_authkit.api.routes.get_backend", lambda name, cfg: be)

    if session_store == "redis":
        redis_url = os.environ[REDIS_URL_ENV]
        session_backend = RedisBackend(redis_url, config.SESSION_MAX_AGE)
        session_backend.client.flushdb()
    else:
        redis_url = None
        # Fresh instance so the DB-backed version getter is resolved for "sql".
        session_backend = CookieBackend(config.SECRET_KEY, config.SESSION_MAX_AGE)
    monkeypatch.setattr(sessions, "backend", session_backend)

    # The limiter follows the session store, as it does in a real deployment.
    limiter = RateLimiter(
        max_attempts=MAX_ATTEMPTS, window_seconds=300, enabled=True, redis_url=redis_url
    )
    monkeypatch.setattr("dagster_authkit.auth.rate_limiter._rate_limiter", limiter)

    yield be

    if session_store == "redis":
        session_backend.client.flushdb()
        session_backend.client.close()
    UserTable.drop_table(safe=True)
    be.db.close()


@pytest.fixture
def app(backend):
    """Minimal ASGI app: auth routes + one protected page behind the middleware."""

    async def home(request):
        return PlainTextResponse(f"hello {request.state.user.username}")

    return Starlette(
        routes=[
            Mount("/auth", routes=create_auth_routes().routes),
            Route("/", home),
        ],
        middleware=[Middleware(DagsterAuthMiddleware)],
    )


def make_client(app) -> TestClient:
    # https so that Secure cookies are stored and sent back by the client.
    return TestClient(app, base_url="https://testserver", follow_redirects=False)


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "CSRF token not found in page"
    return match.group(1)


def login(client: TestClient, username: str, password: str):
    page = client.get("/auth/login")
    return client.post(
        "/auth/process",
        data={
            "username": username,
            "password": password,
            "csrf_token": csrf_from(page.text),
            "next": "/",
        },
    )


def can_login(app, username: str, password: str) -> bool:
    """Try the credentials on a fresh client, outside the login rate limit."""
    limiter = get_rate_limiter()
    limiter.reset(username)
    limiter.reset("ip:testclient")
    response = login(make_client(app), username, password)
    assert response.status_code == 302
    return "error=" not in response.headers["location"]


def submit(client: TestClient, current=OLD, new=NEW, confirm=None, **extra):
    """Open the form (to obtain the CSRF token) and submit it."""
    page = client.get(URL)
    assert page.status_code == 200
    data = {
        "csrf_token": csrf_from(page.text),
        "current_password": current,
        "new_password": new,
        "new_password_confirm": new if confirm is None else confirm,
    }
    data.update(extra)
    return client.post(URL, data=data)


def row(username: str) -> dict:
    """Full database record of a user."""
    return dict(UserTable.select().where(UserTable.username == username).dicts().get())


def all_rows() -> dict:
    return {r["username"]: r for r in UserTable.select().dicts()}


def is_signed_in(client: TestClient) -> bool:
    return client.get("/").status_code == 200


def expire_version_cache():
    """Simulate the session version cache TTL running out (cookie sessions only)."""
    cache = getattr(sessions.backend, "_version_cache", None)
    if cache is not None:
        cache.clear()


class _AuditCollector(logging.Handler):
    """Collects raw audit log lines; ``events()`` parses them as JSON."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def events(self) -> list:
        return [json.loads(line) for line in self.lines]

    def clear(self):
        self.lines.clear()


@pytest.fixture
def audit():
    """Audit events emitted during the test (the audit logger does not propagate)."""
    collector = _AuditCollector()
    audit_logger.addHandler(collector)
    yield collector
    audit_logger.removeHandler(collector)


@pytest.fixture
def client(app):
    c = make_client(app)
    assert login(c, "alice", OLD).headers["location"] == "/"
    return c


class TestAccess:
    """Who may reach the form at all."""

    def test_get_without_session_redirects_to_login(self, app):
        response = make_client(app).get(URL)
        assert response.status_code == 302
        assert response.headers["location"] == "/auth/login?next=/auth/change-password"

    def test_post_without_session_redirects_and_changes_nothing(self, app):
        before = all_rows()
        token = _generate_csrf_token()
        c = make_client(app)
        c.cookies.set("csrf_token", token)
        response = c.post(
            URL,
            data={
                "csrf_token": token,
                "current_password": OLD,
                "new_password": NEW,
                "new_password_confirm": NEW,
            },
        )
        assert response.status_code == 302
        assert response.headers["location"] == "/auth/login?next=/auth/change-password"
        assert all_rows() == before

    def test_garbage_session_cookie_is_rejected(self, app):
        c = make_client(app)
        c.cookies.set(config.SESSION_COOKIE_NAME, "not-a-session")
        assert c.get(URL).status_code == 302
        assert c.post(URL, data={}).status_code == 302

    def test_session_signed_with_other_key_is_rejected(self, app):
        forged = CookieBackend("another-secret-key-0000", 3600).serializer.dumps(
            {"username": "alice", "role": "VIEWER", "_v": 1}
        )
        c = make_client(app)
        c.cookies.set(config.SESSION_COOKIE_NAME, forged)
        assert c.get(URL).status_code == 302

    def test_login_redirect_target_leads_back_to_form(self, app):
        """After signing in via the redirect, the user lands on the form."""
        c = make_client(app)
        location = c.get(URL).headers["location"]
        page = c.get(location)
        assert 'name="next" value="/auth/change-password"' in page.text
        response = c.post(
            "/auth/process",
            data={
                "username": "alice",
                "password": OLD,
                "csrf_token": csrf_from(page.text),
                "next": "/auth/change-password",
            },
        )
        assert response.headers["location"] == "/auth/change-password"

    def test_form_is_rendered_for_signed_in_user(self, client):
        response = client.get(URL)
        assert response.status_code == 200
        body = response.text
        assert 'action="/auth/change-password"' in body
        assert "Signed in as alice" in body
        for field in ("current_password", "new_password", "new_password_confirm"):
            assert f'name="{field}"' in body
        assert "success-message" not in body.split("</style>")[1]
        assert "error-message" not in body.split("</style>")[1]

    def test_form_offers_no_other_editable_fields(self, client):
        """Only the three password inputs and the CSRF token are submitted."""
        names = re.findall(r'<input[^>]*\sname="([^"]+)"', client.get(URL).text)
        assert sorted(names) == [
            "csrf_token",
            "current_password",
            "new_password",
            "new_password_confirm",
        ]

    def test_form_is_not_cacheable(self, client):
        assert client.get(URL).headers["cache-control"] == "no-store"

    def test_deactivated_user_cannot_change_password(self, client, backend):
        token = csrf_from(client.get(URL).text)
        backend.delete_user("alice")
        expire_version_cache()
        before = all_rows()

        response = client.post(
            URL,
            data={
                "csrf_token": token,
                "current_password": OLD,
                "new_password": NEW,
                "new_password_confirm": NEW,
            },
        )

        # Normally the session is already revoked (302 to login). Should a
        # session outlive the deactivation, the current-password check still
        # refuses the user (400).
        assert response.status_code in (302, 400)
        assert all_rows() == before
        assert backend.verify_password("alice", NEW) is False

    @pytest.mark.parametrize("method", ["put", "delete", "patch"])
    def test_other_methods_not_allowed(self, client, method):
        assert getattr(client, method)(URL).status_code == 405


class TestSuccessfulChange:
    """Happy path and its side effects."""

    def test_redirects_back_with_success_message(self, client):
        response = submit(client)
        assert response.status_code == 303
        assert response.headers["location"] == "/auth/change-password?success=1"

        page = client.get(response.headers["location"])
        assert page.status_code == 200
        assert "Password changed successfully." in page.text

    def test_new_password_works_and_old_does_not(self, client, app):
        submit(client)
        assert can_login(app, "alice", NEW) is True
        assert can_login(app, "alice", OLD) is False

    def test_only_password_and_session_version_change(self, client):
        """Nothing but the hash (and the session version) may be modified."""
        before = row("alice")
        assert submit(client).status_code == 303
        after = row("alice")

        assert after["password_hash"] != before["password_hash"]
        assert after["session_version"] == before["session_version"] + 1

        untouched = set(before) - {"password_hash", "session_version"}
        assert untouched == {
            "username",
            "role_value",
            "email",
            "full_name",
            "is_active",
            "created_at",
            "last_login",
        }
        for column in untouched:
            assert after[column] == before[column], f"{column} must not change"

    def test_other_users_are_untouched(self, client, app):
        before = row("bob")
        assert submit(client).status_code == 303
        assert row("bob") == before
        assert can_login(app, "bob", BOB_PASSWORD) is True

    def test_no_user_is_created_or_removed(self, client):
        assert submit(client).status_code == 303
        assert sorted(all_rows()) == ["alice", "bob"]

    def test_stored_hash_matches_new_password_only(self, client, backend):
        submit(client)
        assert backend.verify_password("alice", NEW) is True
        assert backend.verify_password("alice", OLD) is False
        assert NEW not in row("alice")["password_hash"]

    def test_current_browser_stays_signed_in(self, client):
        old_cookie = client.cookies.get(config.SESSION_COOKIE_NAME)
        response = submit(client)
        assert response.status_code == 303

        new_cookie = client.cookies.get(config.SESSION_COOKIE_NAME)
        assert new_cookie and new_cookie != old_cookie
        assert is_signed_in(client)

        # Still valid once the cached session version is re-read from the DB.
        expire_version_cache()
        assert is_signed_in(client)
        assert client.get("/").text == "hello alice"

    def test_new_session_cookie_attributes(self, client):
        response = submit(client)
        header = next(
            h
            for h in response.headers.get_list("set-cookie")
            if h.startswith(f"{config.SESSION_COOKIE_NAME}=")
        )
        assert "HttpOnly" in header
        assert f"Max-Age={config.SESSION_MAX_AGE}" in header
        assert f"samesite={config.SESSION_COOKIE_SAMESITE}" in header.lower()
        assert ("Secure" in header) is config.SESSION_COOKIE_SECURE

    def test_new_session_keeps_identity_and_role(self, client):
        submit(client)
        data = sessions.validate(client.cookies.get(config.SESSION_COOKIE_NAME))
        assert data["username"] == "alice"
        assert AuthUser.from_dict(data).role == Role.VIEWER
        assert data["email"] == "alice@example.com"
        assert data["full_name"] == "Alice A"

    def test_previous_session_cookie_is_revoked(self, client, app):
        old_cookie = client.cookies.get(config.SESSION_COOKIE_NAME)
        submit(client)

        stale = make_client(app)
        stale.cookies.set(config.SESSION_COOKIE_NAME, old_cookie)
        assert not is_signed_in(stale)
        expire_version_cache()
        assert not is_signed_in(stale)

    def test_other_devices_are_signed_out(self, client, app):
        other_device = make_client(app)
        login(other_device, "alice", OLD)
        assert is_signed_in(other_device)

        assert submit(client).status_code == 303

        assert not is_signed_in(other_device)
        assert is_signed_in(client)

    def test_other_users_sessions_survive(self, client, app):
        bob = make_client(app)
        login(bob, "bob", BOB_PASSWORD)
        submit(client)
        expire_version_cache()
        assert is_signed_in(bob)

    def test_can_change_again_with_new_session(self, client, app):
        """The rotated session and its CSRF cookie remain fully usable."""
        assert submit(client, current=OLD, new=NEW).status_code == 303
        assert submit(client, current=NEW, new="ThirdPassw0rd").status_code == 303
        assert can_login(app, "alice", "ThirdPassw0rd") is True
        assert can_login(app, "alice", NEW) is False

    @pytest.mark.parametrize("role", [Role.VIEWER, Role.LAUNCHER, Role.EDITOR, Role.ADMIN])
    def test_every_role_may_change_own_password(self, app, backend, role):
        """The REST write-role restriction must not block read-only users."""
        backend.add_user("carol", OLD, role)
        c = make_client(app)
        login(c, "carol", OLD)
        assert submit(c).status_code == 303
        assert can_login(app, "carol", NEW) is True
        assert row("carol")["role_value"] == role.value

    @pytest.mark.parametrize(
        "password",
        [
            "Aa1 with spaces",
            "Žluťoučký kůň 1",
            "Aa1<script>alert(1)</script>",
            "Aa1'; DROP TABLE users; --",
            "Aa1" + "x" * 253,  # exactly the maximum length
        ],
    )
    def test_unusual_but_valid_passwords(self, client, app, password):
        assert submit(client, new=password).status_code == 303
        assert can_login(app, "alice", password) is True
        assert sorted(all_rows()) == ["alice", "bob"]

    def test_new_password_is_not_trimmed(self, client, app):
        padded = " Passw0rd "
        assert submit(client, new=padded).status_code == 303
        assert can_login(app, "alice", padded) is True
        assert can_login(app, "alice", padded.strip()) is False


class TestOnlyOwnAccount:
    """The target account always comes from the session."""

    @pytest.mark.parametrize("field", ["username", "user", "target", "performed_by"])
    def test_extra_form_fields_cannot_redirect_the_change(self, client, app, field):
        bob_before = row("bob")
        response = submit(client, **{field: "bob"})
        assert response.status_code == 303
        assert row("bob") == bob_before
        assert can_login(app, "bob", BOB_PASSWORD) is True
        assert can_login(app, "alice", NEW) is True

    def test_query_string_cannot_select_another_user(self, client, app):
        bob_before = row("bob")
        page = client.get(URL)
        response = client.post(
            URL + "?username=bob",
            data={
                "csrf_token": csrf_from(page.text),
                "current_password": OLD,
                "new_password": NEW,
                "new_password_confirm": NEW,
            },
        )
        assert response.status_code == 303
        assert row("bob") == bob_before

    def test_knowing_another_users_password_does_not_help(self, client, app):
        """Bob's password is not accepted as alice's current password."""
        before = all_rows()
        response = submit(client, current=BOB_PASSWORD)
        assert response.status_code == 400
        assert all_rows() == before

    def test_extra_fields_cannot_change_profile_or_role(self, client):
        before = row("alice")
        submit(
            client,
            role="ADMIN",
            role_value="40",
            email="evil@example.com",
            full_name="Mallory",
            is_active="0",
        )
        after = row("alice")
        for column in ("role_value", "email", "full_name", "is_active"):
            assert after[column] == before[column]


class TestRejectedChange:
    """Every rejected submission must leave the database untouched."""

    def assert_rejected(self, client, app, response, status, message):
        assert response.status_code == status
        assert message in response.text
        assert "error-message" in response.text.split("</style>")[1]
        assert can_login(app, "alice", OLD) is True
        assert can_login(app, "alice", NEW) is False

    def test_wrong_current_password(self, client, app):
        before = all_rows()
        response = submit(client, current="WrongPassw0rd")
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "Current password is incorrect.")

    def test_empty_current_password(self, client, app):
        before = all_rows()
        response = submit(client, current="")
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "All fields are required.")

    def test_empty_new_password(self, client, app):
        before = all_rows()
        response = submit(client, new="", confirm="")
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "All fields are required.")

    def test_confirmation_mismatch(self, client, app):
        before = all_rows()
        response = submit(client, new=NEW, confirm=NEW + "x")
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "New passwords do not match.")

    def test_missing_confirmation(self, client, app):
        before = all_rows()
        response = submit(client, new=NEW, confirm="")
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "New passwords do not match.")

    def test_confirmation_differs_only_in_case(self, client, app):
        before = all_rows()
        response = submit(client, new=NEW, confirm=NEW.lower())
        assert all_rows() == before
        self.assert_rejected(client, app, response, 400, "New passwords do not match.")

    @pytest.mark.parametrize(
        "weak,message",
        [
            ("Aa1", "at least 8 characters"),
            ("Abcdef1", "at least 8 characters"),
            ("alllowercase1", "uppercase letter"),
            ("ALLUPPERCASE1", "lowercase letter"),
            ("NoDigitsHere", "digit"),
            ("12345678", "lowercase letter"),
            ("Aa1" + "x" * 254, "at most 256 characters"),
        ],
    )
    def test_weak_new_password(self, client, app, weak, message):
        before = all_rows()
        response = submit(client, new=weak)
        assert all_rows() == before
        assert response.status_code == 400
        assert message in response.text
        assert can_login(app, "alice", OLD) is True
        assert can_login(app, "alice", weak) is False

    def test_min_length_is_configurable(self, client, app, monkeypatch):
        monkeypatch.setattr(config, "PASSWORD_MIN_LENGTH", 12)
        assert "At least 12 characters" in client.get(URL).text

        response = submit(client, new=NEW)  # 11 characters
        assert response.status_code == 400
        assert "at least 12 characters" in response.text
        assert can_login(app, "alice", OLD) is True

        assert submit(client, new="LongerPassw0rd").status_code == 303

    def test_new_password_same_as_current(self, client, app):
        before = all_rows()
        response = submit(client, current=OLD, new=OLD)
        assert all_rows() == before
        assert response.status_code == 400
        assert "must differ from the current one" in response.text
        assert can_login(app, "alice", OLD) is True

    def test_rejected_submission_keeps_session(self, client):
        cookie = client.cookies.get(config.SESSION_COOKIE_NAME)
        submit(client, current="WrongPassw0rd")
        assert client.cookies.get(config.SESSION_COOKIE_NAME) == cookie
        assert is_signed_in(client)

    def test_rejected_submission_does_not_sign_out_other_devices(self, client, app):
        other_device = make_client(app)
        login(other_device, "alice", OLD)
        submit(client, current="WrongPassw0rd")
        expire_version_cache()
        assert is_signed_in(other_device)

    def test_error_page_does_not_echo_passwords(self, client):
        response = submit(client, current="WrongPassw0rd", new="Sup3rSecretNew", confirm="other")
        assert "WrongPassw0rd" not in response.text
        assert "Sup3rSecretNew" not in response.text

    def test_can_retry_from_error_page(self, client, app):
        """The form re-rendered with an error carries a usable CSRF token."""
        failed = submit(client, current="WrongPassw0rd")
        response = client.post(
            URL,
            data={
                "csrf_token": csrf_from(failed.text),
                "current_password": OLD,
                "new_password": NEW,
                "new_password_confirm": NEW,
            },
        )
        assert response.status_code == 303
        assert can_login(app, "alice", NEW) is True

    def test_success_flag_in_url_does_not_fake_a_change(self, client, app):
        """?success=1 only shows a message; it never changes anything."""
        before = all_rows()
        page = client.get(URL + "?success=1")
        assert page.status_code == 200
        assert all_rows() == before

    def test_success_flag_requires_session(self, app):
        assert make_client(app).get(URL + "?success=1").status_code == 302


class TestCsrf:
    """State-changing POST must carry a matching, signed CSRF token."""

    def form(self, token):
        return {
            "csrf_token": token,
            "current_password": OLD,
            "new_password": NEW,
            "new_password_confirm": NEW,
        }

    def assert_blocked(self, app, response, before):
        assert response.status_code == 403
        assert "Invalid request" in response.text
        assert all_rows() == before
        assert can_login(app, "alice", OLD) is True

    def test_missing_token(self, client, app):
        before = all_rows()
        client.get(URL)
        self.assert_blocked(app, client.post(URL, data=self.form("")), before)

    def test_token_not_matching_cookie(self, client, app):
        before = all_rows()
        client.get(URL)
        response = client.post(URL, data=self.form(_generate_csrf_token()))
        self.assert_blocked(app, response, before)

    def test_valid_token_without_cookie(self, client, app):
        """A correctly signed token alone is not enough (cross-site POST)."""
        before = all_rows()
        token = csrf_from(client.get(URL).text)
        client.cookies.delete("csrf_token")
        self.assert_blocked(app, client.post(URL, data=self.form(token)), before)

    def test_unsigned_token_matching_cookie(self, client, app):
        before = all_rows()
        client.cookies.delete("csrf_token")
        client.cookies.set("csrf_token", "forged")
        self.assert_blocked(app, client.post(URL, data=self.form("forged")), before)

    def test_expired_token(self, client, app, monkeypatch):
        before = all_rows()
        token = csrf_from(client.get(URL).text)
        with monkeypatch.context() as m:
            m.setattr("dagster_authkit.api.routes._CSRF_MAX_AGE", -1)
            response = client.post(URL, data=self.form(token))
        self.assert_blocked(app, response, before)

    def test_csrf_failure_keeps_session(self, client):
        client.get(URL)
        client.post(URL, data=self.form(""))
        assert is_signed_in(client)

    def test_page_reuses_valid_csrf_cookie(self, client):
        """Opening the form twice must not invalidate the first tab's token."""
        first = csrf_from(client.get(URL).text)
        second_response = client.get(URL)
        assert csrf_from(second_response.text) == first
        assert not any(
            h.startswith("csrf_token=") for h in second_response.headers.get_list("set-cookie")
        )

    def test_page_mints_csrf_cookie_when_missing(self, client):
        client.cookies.delete("csrf_token")
        response = client.get(URL)
        header = next(
            h for h in response.headers.get_list("set-cookie") if h.startswith("csrf_token=")
        )
        assert "HttpOnly" in header
        assert header.split(";")[0].split("=", 1)[1] == csrf_from(response.text)


class TestRateLimit:
    """Guessing the current password from a live session is throttled."""

    def test_blocks_after_max_attempts(self, client, app):
        before = all_rows()
        for _ in range(ALLOWED_GUESSES):
            assert submit(client, current="WrongPassw0rd").status_code == 400

        # Even the correct password is refused while limited.
        response = submit(client, current=OLD)
        assert response.status_code == 429
        assert "Too many attempts" in response.text
        assert all_rows() == before
        assert can_login(app, "alice", OLD) is True

    def test_limit_is_per_user(self, client, app):
        for _ in range(MAX_ATTEMPTS + 1):
            submit(client, current="WrongPassw0rd")

        bob = make_client(app)
        login(bob, "bob", BOB_PASSWORD)
        assert submit(bob, current=BOB_PASSWORD).status_code == 303

    def test_limit_does_not_lock_login(self, client, app):
        """Separate limiter key: a throttled form must not block sign-in."""
        for _ in range(MAX_ATTEMPTS + 1):
            submit(client, current="WrongPassw0rd")
        response = login(make_client(app), "alice", OLD)
        assert response.headers["location"] == "/"

    def test_form_validation_errors_do_not_consume_attempts(self, client):
        for _ in range(MAX_ATTEMPTS + 2):
            assert submit(client, new=NEW, confirm="mismatch").status_code == 400
            assert submit(client, new="weak").status_code == 400
        assert submit(client).status_code == 303

    def test_success_resets_counter(self, client):
        for _ in range(ALLOWED_GUESSES - 1):
            submit(client, current="WrongPassw0rd")
        assert submit(client, current=OLD, new=NEW).status_code == 303

        for _ in range(ALLOWED_GUESSES - 1):
            assert submit(client, current="WrongPassw0rd").status_code == 400
        assert submit(client, current=NEW, new="ThirdPassw0rd").status_code == 303

    def test_violation_is_audited(self, client, audit):
        for _ in range(MAX_ATTEMPTS + 1):
            submit(client, current="WrongPassw0rd")
        events = audit.events()
        assert any(e.get("subtype") == "RATE_LIMIT_EXCEEDED" for e in events)
        assert any(
            e.get("event_type") == "PASSWORD_CHANGE_FAILED" and e.get("reason") == "RATE_LIMIT"
            for e in events
        )


class TestUnsupportedBackends:
    """Only the local SQL backend can change passwords."""

    @pytest.mark.parametrize("name", ["dummy", "ldap", "oauth", "proxy"])
    def test_routes_return_404(self, client, app, monkeypatch, name):
        before = all_rows()
        token = csrf_from(client.get(URL).text)
        monkeypatch.setattr(config, "AUTH_BACKEND", name)

        assert client.get(URL).status_code == 404
        response = client.post(
            URL,
            data={
                "csrf_token": token,
                "current_password": OLD,
                "new_password": NEW,
                "new_password_confirm": NEW,
            },
        )
        assert response.status_code == 404
        assert all_rows() == before

    def test_sqlite_alias_is_supported(self, client, app, monkeypatch):
        monkeypatch.setattr(config, "AUTH_BACKEND", "sqlite")
        assert submit(client).status_code == 303
        assert can_login(app, "alice", NEW) is True


class TestSessionStoreFailure:
    """The session store fails after the password was already changed."""

    @pytest.fixture
    def broken_rotation(self, backend, monkeypatch):
        def boom(user_data):
            raise ConnectionError("session store down")

        monkeypatch.setattr(sessions, "rotate", boom)

    def test_user_is_sent_to_login_instead_of_error_500(self, client, broken_rotation):
        response = submit(client)
        assert response.status_code == 303
        location = response.headers["location"]
        assert location.startswith("/auth/login?")
        assert "Password+changed" in location

        page = client.get(location)
        assert page.status_code == 200
        assert "Password changed. Please sign in again." in page.text

    def test_password_is_changed(self, client, app, broken_rotation):
        submit(client)
        assert can_login(app, "alice", NEW) is True
        assert can_login(app, "alice", OLD) is False

    def test_stale_session_cookie_is_cleared(self, client, broken_rotation):
        response = submit(client)
        header = next(
            h
            for h in response.headers.get_list("set-cookie")
            if h.startswith(f"{config.SESSION_COOKIE_NAME}=")
        )
        assert "Max-Age=0" in header
        assert not client.cookies.get(config.SESSION_COOKIE_NAME)

    def test_failure_is_audited(self, client, audit, broken_rotation):
        audit.clear()
        submit(client)
        types = [e["event_type"] for e in audit.events()]
        assert "PASSWORD_CHANGED" in types
        assert "SESSION_ROTATION_FAILED" in types
        assert "SESSION_CREATED" not in types

    def test_error_details_are_not_shown(self, client, broken_rotation):
        response = submit(client)
        assert "session store down" not in response.headers["location"]


class TestBackendFailures:
    """Backend problems surface as an error without side effects."""

    def test_verify_raises(self, client, app, backend, monkeypatch):
        before = all_rows()

        def boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(backend, "verify_password", boom)
        response = submit(client)
        assert response.status_code == 500
        assert "System error" in response.text
        assert "db down" not in response.text
        assert all_rows() == before

    def test_change_raises(self, client, app, backend, monkeypatch):
        before = all_rows()

        def boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(backend, "change_password", boom)
        response = submit(client)
        assert response.status_code == 500
        assert all_rows() == before
        assert is_signed_in(client)

    def test_change_returns_false(self, client, app, backend, monkeypatch):
        before = all_rows()
        monkeypatch.setattr(backend, "change_password", lambda *a, **k: False)
        response = submit(client)
        assert response.status_code == 500
        assert all_rows() == before
        assert can_login(app, "alice", OLD) is True

    def test_backend_called_with_session_user(self, client, backend, monkeypatch):
        """change_password receives the session's username as target and actor."""
        calls = []
        original = backend.change_password

        def spy(username, new_password, performed_by="system"):
            calls.append((username, new_password, performed_by))
            return original(username, new_password, performed_by=performed_by)

        monkeypatch.setattr(backend, "change_password", spy)
        submit(client, username="bob")
        assert calls == [("alice", NEW, "alice")]


class TestAuditAndLogs:
    """Audit trail of the change and secrecy of the passwords."""

    def test_success_is_audited_as_self_service(self, client, audit):
        audit.clear()
        submit(client)
        events = audit.events()

        changed = [e for e in events if e.get("event_type") == "PASSWORD_CHANGED"]
        assert len(changed) == 1
        assert changed[0]["performed_by"] == "alice"
        assert changed[0]["target"] == "alice"
        assert any(
            e.get("event_type") == "SESSION_CREATED" and e.get("performed_by") == "alice"
            for e in events
        )
        assert not any(e.get("event_type") == "PASSWORD_CHANGE_FAILED" for e in events)

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            ({"current": "WrongPassw0rd"}, "INVALID_CURRENT_PASSWORD"),
            ({"confirm": "different"}, "CONFIRMATION_MISMATCH"),
            ({"new": "weak"}, "WEAK_PASSWORD"),
            ({"new": OLD}, "SAME_PASSWORD"),
            ({"current": ""}, "MISSING_FIELDS"),
        ],
    )
    def test_failure_is_audited_with_reason(self, client, audit, kwargs, reason):
        audit.clear()
        submit(client, **kwargs)
        events = audit.events()

        failed = [e for e in events if e.get("event_type") == "PASSWORD_CHANGE_FAILED"]
        assert len(failed) == 1
        assert failed[0]["performed_by"] == "alice"
        assert failed[0]["reason"] == reason
        assert not any(e.get("event_type") == "PASSWORD_CHANGED" for e in events)

    def test_passwords_never_reach_logs(self, client, audit, capfd, caplog):
        secrets_used = ["Curr3ntGuess!", "BrandNewPassw0rd!", "Mismatch3dConfirm"]
        with caplog.at_level("DEBUG"):
            submit(client, current=secrets_used[0], new=secrets_used[1])
            submit(client, new=secrets_used[1], confirm=secrets_used[2])
            submit(client, current=OLD, new=secrets_used[1])

        assert any(e["event_type"] == "PASSWORD_CHANGED" for e in audit.events())
        captured = capfd.readouterr()
        haystack = captured.out + captured.err + caplog.text + "\n".join(audit.lines)
        for secret in secrets_used + [OLD]:
            assert secret not in haystack
