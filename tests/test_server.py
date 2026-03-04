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


class TestShadowMode:
    def test_shadow_returns_allow(self, client):
        resp = client.post("/v1/classify?mode=shadow", json={
            "messages": [{"role": "user", "content": "What are mortgage rates?"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "allow"

    def test_shadow_header_has_real_decision(self, client):
        resp = client.post("/v1/classify?mode=shadow", json={
            "messages": [{"role": "user", "content": "What are mortgage rates?"}]
        })
        assert "x-classification-shadow" in resp.headers
        assert resp.headers["x-classification-shadow"] in ["allow", "deny", "abstain"]

    def test_enforce_mode_default(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 200
        assert "x-classification-shadow" not in resp.headers


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client):
        resp = client.post("/v1/feedback", json={
            "query": "What are mortgage rates?",
            "expected_decision": "allow",
            "actual_decision": "deny",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_feedback_with_notes(self, client):
        resp = client.post("/v1/feedback", json={
            "query": "HSA contribution limits",
            "expected_decision": "allow",
            "actual_decision": "abstain",
            "notes": "This is clearly a financial question",
        })
        assert resp.status_code == 200

    def test_feedback_invalid_decision(self, client):
        resp = client.post("/v1/feedback", json={
            "query": "test",
            "expected_decision": "invalid",
            "actual_decision": "allow",
        })
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_after_classify(self, client):
        client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "test"}]
        })
        resp = client.get("/metrics")
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
