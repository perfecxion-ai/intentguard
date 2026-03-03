"""Tests for API endpoints."""


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["policy_loaded"] is True
        assert data["vertical"] == "finance"

    def test_health_reports_stub_model(self, client):
        resp = client.get("/health")
        data = resp.json()
        # Stub classifier means model_loaded is false
        assert data["model_loaded"] is False


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "intentguard-finance"


class TestClassifyEndpoint:
    def test_basic_classify(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "What are mortgage rates?"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] in ["allow", "deny", "abstain"]
        assert 0 <= data["confidence"] <= 1
        assert data["vertical"] == "finance"

    def test_decision_header(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert "x-classification-decision" in resp.headers
        assert resp.headers["x-classification-decision"] in ["allow", "deny", "abstain"]

    def test_latency_header(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert "x-classification-latency-ms" in resp.headers
        latency = float(resp.headers["x-classification-latency-ms"])
        assert latency >= 0

    def test_no_user_message_returns_400(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "system", "content": "You are a bot."}]
        })
        assert resp.status_code == 400

    def test_empty_messages_returns_422(self, client):
        resp = client.post("/v1/classify", json={"messages": []})
        # pydantic allows empty list but last_user_message returns None -> 400
        # Actually this depends on whether there's a user message
        assert resp.status_code == 400

    def test_multiple_messages_uses_last_user(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow up question"},
            ]
        })
        assert resp.status_code == 200


class TestChatCompletionsEndpoint:
    def test_basic_completion(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "intentguard",
            "messages": [{"role": "user", "content": "What is compound interest?"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_response_has_id(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "intentguard",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")

    def test_no_user_message_returns_400(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "intentguard",
            "messages": [{"role": "system", "content": "System prompt"}]
        })
        assert resp.status_code == 400
