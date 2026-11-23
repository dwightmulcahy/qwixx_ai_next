"""Regression checks for secure-cookie and error disclosure policies."""
from pathlib import Path

from qwixx_ai import distributed


def test_cookie_secure_flag_does_not_trust_forwarded_proto():
    source = Path(distributed.__file__).read_text()
    assert "self.headers.get('X-Forwarded-Proto','').lower()=='https'" not in source
    assert "os.getenv('QWIXX_COOKIE_SECURE','false').lower()=='true'" in source


def test_internal_get_errors_are_not_returned_to_clients():
    source = Path(distributed.__file__).read_text()
    assert "logging.exception('Unhandled HTTP GET error')" in source
    assert "self.reply({'error':'Internal server error'},500)" in source
    assert "except Exception as e:self.reply({'error':str(e)},500)" not in source
