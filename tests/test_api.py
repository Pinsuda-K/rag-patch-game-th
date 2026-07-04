from fastapi.testclient import TestClient
from src.serve.api import app

def test_contract():
    c = TestClient(app)
    r = c.post("/query", json={"q":"ทดสอบ","use_reranker":False,"k":5})
    assert r.status_code==200
    body=r.json()
    assert "text" in body and "citations" in body and "timings" in body
