"""Tests for the dashboard/API auth manager."""

from __future__ import annotations

import time

from thermalsentry.config import SecuritySettings
from thermalsentry.web.security import SECURITY_HEADERS, AuthManager


def _mgr(**kw) -> AuthManager:
    base = dict(
        api_key="testkey",
        basic_auth_user="admin",
        basic_auth_password="pw",
        session_secret="s3cr3t",
        session_ttl_seconds=3600,
    )
    base.update(kw)
    return AuthManager(SecuritySettings(**base))


def test_check_api_key():
    auth = _mgr()
    assert auth.check_api_key("testkey") is True
    assert auth.check_api_key("wrong") is False
    assert auth.check_api_key(None) is False
    assert auth.check_api_key("") is False


def test_check_basic():
    auth = _mgr()
    assert auth.check_basic("admin", "pw") is True
    assert auth.check_basic("admin", "bad") is False
    assert auth.check_basic("nobody", "pw") is False
    assert auth.check_basic(None, "pw") is False
    assert auth.check_basic("admin", None) is False


def test_session_roundtrip():
    auth = _mgr()
    token = auth.issue_session("admin")
    assert auth.verify_session(token) is True


def test_session_rejects_none_and_garbage():
    auth = _mgr()
    assert auth.verify_session(None) is False
    assert auth.verify_session("") is False
    assert auth.verify_session("not-a-token") is False


def test_session_rejects_tampered_signature():
    auth = _mgr()
    token = auth.issue_session("admin")
    user, expiry, _sig = token.rsplit(".", 2)
    tampered = f"{user}.{expiry}.deadbeef"
    assert auth.verify_session(tampered) is False


def test_session_rejects_tampered_payload():
    auth = _mgr()
    token = auth.issue_session("admin")
    _user, expiry, sig = token.rsplit(".", 2)
    # Re-sign nothing; change the user but keep old signature -> invalid.
    forged = f"attacker.{expiry}.{sig}"
    assert auth.verify_session(forged) is False


def test_session_expired():
    auth = _mgr(session_ttl_seconds=-10)
    token = auth.issue_session("admin")
    assert auth.verify_session(token) is False


def test_session_bad_expiry_field():
    auth = _mgr()
    payload = "admin.notanint"
    sig = auth._sign(payload)
    token = f"{payload}.{sig}"
    assert auth.verify_session(token) is False


def test_generated_secret_flags():
    auth = AuthManager(SecuritySettings(api_key=None, basic_auth_password=None))
    assert auth._generated_api_key is True
    assert auth._generated_password is True
    assert auth.api_key  # a random key was generated
    assert auth.basic_password

    fixed = AuthManager(
        SecuritySettings(api_key="k", basic_auth_password="p")
    )
    assert fixed._generated_api_key is False
    assert fixed._generated_password is False


def test_two_managers_have_distinct_secrets():
    a = AuthManager(SecuritySettings(api_key=None, basic_auth_password=None, session_secret=None))
    b = AuthManager(SecuritySettings(api_key=None, basic_auth_password=None, session_secret=None))
    # Different processes -> different random tokens; sessions don't cross-verify.
    token = a.issue_session("admin")
    assert b.verify_session(token) is False
    # And a verifies its own token.
    assert a.verify_session(token) is True


def test_security_headers_present():
    assert "Content-Security-Policy" in SECURITY_HEADERS
    assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"


def test_session_not_expired_now():
    auth = _mgr(session_ttl_seconds=60)
    token = auth.issue_session("admin")
    # Expiry is in the future.
    _user, expiry, _sig = token.rsplit(".", 2)
    assert int(expiry) >= int(time.time())
