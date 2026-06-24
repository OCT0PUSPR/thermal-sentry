"""Security for the dashboard + API: auth, sessions, CORS, headers, rate limiting.

Auth model:
  * API clients send ``X-API-Key: <key>`` (or ``?api_key=``).
  * Browser users log in via HTTP Basic at ``/login`` which sets a signed,
    HMAC session cookie. Subsequent dashboard/WS requests present the cookie.

Secrets (API key, basic-auth password, session secret) come from settings, which
come from the environment. If they are unset a random value is generated at
startup and logged once (dev convenience) -- never a fixed default.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..config import SecuritySettings


class AuthManager:
    """Validates API keys, basic-auth credentials, and signed session cookies."""

    COOKIE_NAME = "ts_session"

    def __init__(self, settings: "SecuritySettings") -> None:
        self.s = settings
        # Resolve (or generate) secrets once.
        self.api_key = settings.api_key or secrets.token_urlsafe(24)
        self.session_secret = (settings.session_secret or secrets.token_urlsafe(32)).encode()
        self.basic_password = settings.basic_auth_password or secrets.token_urlsafe(16)
        self._generated_api_key = settings.api_key is None
        self._generated_password = settings.basic_auth_password is None

    # -- API key --------------------------------------------------------------

    def check_api_key(self, provided: Optional[str]) -> bool:
        if not provided:
            return False
        return hmac.compare_digest(provided, self.api_key)

    # -- basic auth -----------------------------------------------------------

    def check_basic(self, user: Optional[str], password: Optional[str]) -> bool:
        if user is None or password is None:
            return False
        ok_user = hmac.compare_digest(user, self.s.basic_auth_user)
        ok_pass = hmac.compare_digest(password, self.basic_password)
        return ok_user and ok_pass

    # -- session cookies (HMAC-signed) ----------------------------------------

    def issue_session(self, user: str) -> str:
        """Create a signed session token: ``user.expiry.hexsig``."""
        expiry = int(time.time()) + self.s.session_ttl_seconds
        payload = f"{user}.{expiry}"
        sig = self._sign(payload)
        return f"{payload}.{sig}"

    def verify_session(self, token: Optional[str]) -> bool:
        if not token:
            return False
        try:
            user, expiry_s, sig = token.rsplit(".", 2)
        except ValueError:
            return False
        payload = f"{user}.{expiry_s}"
        if not hmac.compare_digest(sig, self._sign(payload)):
            return False
        try:
            return int(expiry_s) >= int(time.time())
        except ValueError:
            return False

    def _sign(self, payload: str) -> str:
        return hmac.new(self.session_secret, payload.encode(), hashlib.sha256).hexdigest()


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    # CSP allows the dashboard's own inline-free assets + websocket to self.
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; "
        "script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws: wss:; frame-ancestors 'none'"
    ),
}
