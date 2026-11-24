"""Browser security policies are applied to actual HTTP response headers."""
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler

from qwixx_ai.distributed import Handler, content_security_policy


def test_csp_restricts_unused_resource_types():
    policy = content_security_policy("<html></html>")
    for directive in ("media-src 'none'", "font-src 'self'", "worker-src 'self'", "manifest-src 'self'"):
        assert directive in policy.split("; ")


def test_browser_capabilities_and_cross_origin_resources_are_restricted():
    handler = object.__new__(Handler)
    handler.send_header = Mock()
    with patch.object(BaseHTTPRequestHandler, "end_headers"):
        handler.end_headers()
    handler.send_header.assert_any_call(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
    )
    handler.send_header.assert_any_call("Cross-Origin-Resource-Policy", "same-origin")
