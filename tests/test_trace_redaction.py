from app.observability.tracing import redact_mapping


def test_sensitive_fields_are_redacted_recursively() -> None:
    value = redact_mapping(
        {
            "api_key": "secret",
            "recipient": "a@b.com",
            "query": "policy",
            "headers": {"Authorization": "Bearer secret"},
            "items": [{"access_token": "token-value"}, {"password": "pw"}],
        }
    )

    assert value == {
        "api_key": "[REDACTED]",
        "recipient": "a@b.com",
        "query": "policy",
        "headers": {"Authorization": "[REDACTED]"},
        "items": [{"access_token": "[REDACTED]"}, {"password": "[REDACTED]"}],
    }
