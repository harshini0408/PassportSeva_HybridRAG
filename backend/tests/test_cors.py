from fastapi.testclient import TestClient

from app.main import app


def test_cors_allows_capacitor_webview_origin():
    c = TestClient(app)
    r = c.options("/api/ask", headers={"Origin": "http://localhost",
                                       "Access-Control-Request-Method": "POST",
                                       "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost"


def test_cors_rejects_unknown_origin():
    c = TestClient(app)
    r = c.options("/api/ask", headers={"Origin": "http://evil.example",
                                       "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
