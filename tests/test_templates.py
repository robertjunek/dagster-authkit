"""
Unit tests for utils/templates.py

Covers:
- render_login_page basic output
- render_login_page XSS escaping
- render_login_page CSRF token injection
- render_login_page error message display
- render_403_page output
- render_403_page XSS escaping
- render_user_menu_injection output
- render_change_password_page output and XSS escaping
"""

import html

from dagster_authkit.auth.backends.base import AuthUser, Role
from dagster_authkit.utils.templates import (
    render_403_page,
    render_change_password_page,
    render_login_page,
    render_user_menu_injection,
)


def make_user(username="testuser", role=Role.VIEWER, email="test@test.com", full_name="Test User"):
    return AuthUser(username=username, role=role, email=email, full_name=full_name)


class TestRenderLoginPage:
    """Verifies the login page template."""

    def test_returns_html_string(self):
        """Should return a non-empty string."""
        result = render_login_page()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_doctype(self):
        """Should start with DOCTYPE."""
        result = render_login_page()
        assert "<!DOCTYPE html>" in result

    def test_contains_form(self):
        """Should contain a login form."""
        result = render_login_page()
        assert '<form method="post"' in result
        assert "username" in result
        assert "password" in result

    def test_default_next_url(self):
        """Default next_url should be /."""
        result = render_login_page()
        assert 'value="/"' in result

    def test_custom_next_url(self):
        """Should include the provided next_url."""
        result = render_login_page(next_url="/workspace/my-pipeline")
        assert 'value="/workspace/my-pipeline"' in result

    def test_escapes_next_url(self):
        """next_url should be HTML-escaped."""
        result = render_login_page(next_url='/?foo=<script>alert(1)</script>')
        safe = html.escape('/?foo=<script>alert(1)</script>', quote=True)
        assert safe in result
        assert '<script>alert(1)</script>' not in result

    def test_no_error_div_when_no_error(self):
        """No error div should be present when error is empty."""
        result = render_login_page(error="")
        assert "<div class='error-message'>" not in result

    def test_renders_error_message(self):
        """Should display error message when provided."""
        result = render_login_page(error="Invalid credentials")
        assert "error-message" in result
        assert "Invalid credentials" in result

    def test_escapes_error_message(self):
        """Error message should be HTML-escaped."""
        result = render_login_page(error='<script>alert("xss")</script>')
        safe = html.escape('<script>alert("xss")</script>', quote=True)
        assert safe in result
        assert '<script>alert("xss")</script>' not in result

    def test_includes_csrf_token_hidden_field(self):
        """CSRF token should be in a hidden input."""
        result = render_login_page(csrf_token="csrf-abc-123")
        assert 'name="csrf_token"' in result
        assert 'value="csrf-abc-123"' in result

    def test_escapes_csrf_token(self):
        """CSRF token should be HTML-escaped."""
        result = render_login_page(csrf_token='<script>bad</script>')
        safe = html.escape('<script>bad</script>', quote=True)
        assert safe in result


class TestRender403Page:
    """Verifies the 403 Forbidden page template."""

    def test_returns_html_string(self):
        """Should return a non-empty string."""
        user = make_user()
        result = render_403_page(user, "/graphql", "POST", "Insufficient role")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_403_title(self):
        """Should display 403 Forbidden."""
        user = make_user()
        result = render_403_page(user, "/graphql", "POST", "Insufficient role")
        assert "403 Forbidden" in result

    def test_shows_username(self):
        """Should display the user's username."""
        user = make_user(username="john_doe")
        result = render_403_page(user, "/path", "GET", "reason")
        assert "john_doe" in result

    def test_shows_role(self):
        """Should display the user's role name."""
        user = make_user(role=Role.VIEWER)
        result = render_403_page(user, "/path", "GET", "reason")
        assert Role.VIEWER.name in result

    def test_shows_method_and_path(self):
        """Should display HTTP method and path."""
        user = make_user()
        result = render_403_page(user, "/workspace/assets", "DELETE", "Not allowed")
        assert "DELETE" in result
        assert "/workspace/assets" in result

    def test_shows_reason(self):
        """Should display denial reason."""
        user = make_user()
        result = render_403_page(user, "/path", "GET", "Requires ADMIN role")
        assert "Requires ADMIN role" in result

    def test_escapes_username(self):
        """Username should be HTML-escaped."""
        user = make_user(username='<script>alert(1)</script>')
        result = render_403_page(user, "/path", "GET", "reason")
        safe = html.escape('<script>alert(1)</script>', quote=True)
        assert safe in result
        assert '<script>alert(1)</script>' not in result

    def test_escapes_path(self):
        """Path should be HTML-escaped."""
        user = make_user()
        result = render_403_page(user, '/path?q=<img onerror=alert(1)>', "GET", "reason")
        safe = html.escape('/path?q=<img onerror=alert(1)>', quote=True)
        assert safe in result

    def test_escapes_reason(self):
        """Reason should be HTML-escaped."""
        user = make_user()
        result = render_403_page(user, "/path", "GET", '<b>bad</b>')
        safe = html.escape('<b>bad</b>', quote=True)
        assert safe in result


class TestRenderUserMenuInjection:
    """Verifies the user menu injection script."""

    def test_returns_html_string(self):
        """Should return a non-empty string."""
        result = render_user_menu_injection('{"username": "test"}', debug=False, safe_mode=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_user_data(self):
        """Should inject the user JSON data."""
        result = render_user_menu_injection('{"full_name": "John"}', debug=False, safe_mode=True)
        assert '{"full_name": "John"}' in result

    def test_contains_logout_link(self):
        """Should include a logout link."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=True)
        assert "/auth/logout" in result

    def test_debug_disabled_by_default(self):
        """When debug is False, DEBUG should be false."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=True)
        assert "const DEBUG = false" in result

    def test_debug_enabled(self):
        """When debug is True, DEBUG should be true."""
        result = render_user_menu_injection('{}', debug=True, safe_mode=True)
        assert "const DEBUG = true" in result

    def test_safe_mode_enabled(self):
        """When safe_mode is True, SAFE_MODE should be true."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=True)
        assert "const SAFE_MODE = true" in result

    def test_safe_mode_disabled(self):
        """When safe_mode is False, SAFE_MODE should be false."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=False)
        assert "const SAFE_MODE = false" in result

    def test_contains_script_tag(self):
        """Should contain a script block."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=True)
        assert "<script>" in result
        assert "</script>" in result

    def test_contains_style_tag(self):
        """Should contain a style block."""
        result = render_user_menu_injection('{}', debug=False, safe_mode=True)
        assert "<style>" in result


class TestRenderChangePasswordPage:
    """Verifies the self-service change password page template."""

    def test_returns_complete_html(self):
        result = render_change_password_page("alice")
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result
        assert "<title>Change password - Dagster</title>" in result

    def test_form_posts_to_change_password(self):
        result = render_change_password_page("alice")
        assert '<form method="post" action="/auth/change-password">' in result

    def test_contains_exactly_three_password_fields(self):
        """Only current password and the new password twice may be entered."""
        import re

        result = render_change_password_page("alice", csrf_token="tok")
        inputs = re.findall(r"<input[^>]*>", result)
        by_name = {re.search(r'name="([^"]+)"', i).group(1): i for i in inputs}
        assert sorted(by_name) == [
            "csrf_token",
            "current_password",
            "new_password",
            "new_password_confirm",
        ]
        for name in ("current_password", "new_password", "new_password_confirm"):
            assert 'type="password"' in by_name[name]
            assert "required" in by_name[name]
        assert 'type="hidden"' in by_name["csrf_token"]

    def test_password_fields_are_never_prefilled(self):
        result = render_change_password_page("alice", csrf_token="tok", error="bad")
        assert result.count("value=") == 1  # the CSRF token only

    def test_autocomplete_hints(self):
        result = render_change_password_page("alice")
        assert result.count('autocomplete="current-password"') == 1
        assert result.count('autocomplete="new-password"') == 2

    def test_csrf_token_injected(self):
        result = render_change_password_page("alice", csrf_token="abc123")
        assert 'name="csrf_token" value="abc123"' in result

    def test_shows_username(self):
        result = render_change_password_page("alice")
        assert "Signed in as alice" in result

    def test_no_message_by_default(self):
        body = render_change_password_page("alice").split("</style>")[1]
        assert "error-message" not in body
        assert "success-message" not in body

    def test_error_message(self):
        body = render_change_password_page("alice", error="Nope").split("</style>")[1]
        assert "<div class='error-message'>Nope</div>" in body
        assert "success-message" not in body

    def test_success_message(self):
        body = render_change_password_page("alice", success=True).split("</style>")[1]
        assert "Password changed successfully." in body
        assert "error-message" not in body

    def test_success_takes_precedence_over_error(self):
        body = render_change_password_page("alice", error="Nope", success=True).split("</style>")[1]
        assert "Password changed successfully." in body
        assert "Nope" not in body

    def test_policy_hint_uses_min_length(self):
        result = render_change_password_page("alice", min_length=12)
        assert "At least 12 characters" in result
        assert 'minlength="12"' in result

    def test_back_link(self):
        result = render_change_password_page("alice")
        assert '<a class="back-link" href="/">' in result

    def test_xss_in_username_escaped(self):
        payload = "<script>alert(1)</script>"
        result = render_change_password_page(payload)
        assert payload not in result
        assert html.escape(payload, quote=True) in result

    def test_xss_in_error_escaped(self):
        payload = '"><img src=x onerror=alert(1)>'
        result = render_change_password_page("alice", error=payload)
        assert payload not in result
        assert html.escape(payload, quote=True) in result

    def test_xss_in_csrf_token_escaped(self):
        payload = '"><script>alert(1)</script>'
        result = render_change_password_page("alice", csrf_token=payload)
        assert payload not in result

    def test_shares_logo_with_login_page(self):
        """Both pages should render exactly one copy of the shared logo."""
        for page in (render_login_page(), render_change_password_page("alice")):
            assert page.count('<svg class="logo-svg"') == 1
            assert page.count("<path ") == 4
            assert "{_AUTH_PAGE_LOGO}" not in page

    def test_shares_stylesheet_with_login_page(self):
        """Both standalone pages should render the same card styling."""
        login_style = render_login_page().split("<style>")[1].split("</style>")[0]
        page_style = render_change_password_page("alice").split("<style>")[1].split("</style>")[0]
        assert login_style == page_style
        assert ".login-card {" in page_style


class TestUserMenuChangePasswordLink:
    """Verifies the change password entry of the injected user menu."""

    def test_link_present_in_sidebar_menu_and_safe_mode(self):
        result = render_user_menu_injection("{}", debug=False, safe_mode=True)
        assert result.count('href="/auth/change-password"') == 2

    def test_link_is_conditional_on_user_flag(self):
        """The link must only render when the backend supports password changes."""
        result = render_user_menu_injection("{}", debug=False, safe_mode=True)
        assert result.count("user.can_change_password ?") == 2

    def test_link_precedes_sign_out(self):
        result = render_user_menu_injection("{}", debug=False, safe_mode=True)
        script = result.split("<script>")[1]
        assert script.index("/auth/change-password") < script.index("/auth/logout")

    def test_logout_link_still_present(self):
        result = render_user_menu_injection("{}", debug=False, safe_mode=True)
        assert result.count('href="/auth/logout"') == 2
