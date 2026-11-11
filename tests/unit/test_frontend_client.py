"""Frontend REST client: configurable URL and friendly errors (TC-UI-*)."""

import importlib


def _client(monkeypatch, url):
    monkeypatch.setenv("BACKEND_URL", url)
    import api_client

    return importlib.reload(api_client)


def test_backend_url_comes_from_env(monkeypatch):
    api = _client(monkeypatch, "http://example:1234/")
    assert api.BACKEND_URL == "http://example:1234"


def test_unreachable_backend_gives_friendly_error(monkeypatch):
    api = _client(monkeypatch, "http://127.0.0.1:9")
    data, err = api.get("/health")
    assert data is None and "Cannot reach" in err


def test_error_codes_map_to_plain_messages(monkeypatch):
    api = _client(monkeypatch, "http://x")

    class Resp:
        status_code = 503

        def json(self):
            return {"error_code": "MODEL_UNAVAILABLE", "message": "raw"}

    assert "starting up" in api._error(Resp())
