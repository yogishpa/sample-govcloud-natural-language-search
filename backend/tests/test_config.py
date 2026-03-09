"""Tests for core configuration module.

Validates Settings class defaults, environment variable loading,
validation constraints, and GovCloud vs commercial ARN prefix logic.

Requirements: 9.1, 9.2, 10.4
"""

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify default values match design doc specifications."""

    def test_default_environment(self):
        s = Settings()
        assert s.environment == "commercial"

    def test_default_embedding_model_id(self):
        s = Settings()
        assert s.embedding_model_id == "amazon.titan-embed-text-v2:0"

    def test_default_session_history_limit(self):
        s = Settings()
        assert s.session_history_limit == 10

    def test_default_rate_limit(self):
        s = Settings()
        assert s.rate_limit_window_seconds == 60
        assert s.rate_limit_max_requests == 20

    def test_default_max_context_tokens(self):
        s = Settings()
        assert s.max_context_tokens == 4096

    def test_default_retry_config(self):
        s = Settings()
        assert s.retry_max_attempts == 3
        assert s.retry_base_delay_seconds == 1.0
        assert s.retry_max_delay_seconds == 8.0

    def test_default_bedrock_region(self):
        s = Settings()
        assert s.bedrock_region == "us-east-1"


class TestArnPrefix:
    """Verify ARN prefix logic for commercial vs GovCloud (Req 9.1, 10.4)."""

    def test_commercial_arn_prefix(self):
        s = Settings(environment="commercial")
        assert s.arn_prefix == "arn:aws"

    def test_govcloud_arn_prefix(self):
        s = Settings(environment="govcloud")
        assert s.arn_prefix == "arn:aws-us-gov"

    def test_non_govcloud_environment_uses_commercial_prefix(self):
        s = Settings(environment="other")
        assert s.arn_prefix == "arn:aws"


class TestEffectiveInferenceProfileArn:
    """Verify inference profile ARN generation (Req 9.2)."""

    def test_explicit_arn_returned_as_is(self):
        explicit = "arn:aws:bedrock:us-east-1:123456:inference-profile/custom"
        s = Settings(inference_profile_arn=explicit)
        assert s.effective_inference_profile_arn == explicit

    def test_commercial_auto_generated_arn(self):
        s = Settings(
            environment="commercial",
            bedrock_region="us-east-1",
            aws_account_id="111222333444",
        )
        expected = (
            "arn:aws:bedrock:us-east-1:111222333444"
            ":inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert s.effective_inference_profile_arn == expected

    def test_govcloud_auto_generated_arn(self):
        s = Settings(
            environment="govcloud",
            aws_account_id="555666777888",
        )
        expected = (
            "arn:aws-us-gov:bedrock:us-gov-east-1:555666777888"
            ":inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert s.effective_inference_profile_arn == expected

    def test_auto_generated_arn_placeholder_without_account(self):
        s = Settings(environment="commercial", bedrock_region="us-west-2")
        assert "{account}" in s.effective_inference_profile_arn
        assert "us-west-2" in s.effective_inference_profile_arn


class TestEnvironmentVariableLoading:
    """Verify settings load from environment variables."""

    def test_opensearch_endpoint_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://my-collection.aoss.amazonaws.com")
        s = Settings()
        assert s.opensearch_endpoint == "https://my-collection.aoss.amazonaws.com"

    def test_bedrock_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_REGION", "us-gov-east-1")
        s = Settings()
        assert s.bedrock_region == "us-gov-east-1"

    def test_session_history_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_HISTORY_LIMIT", "20")
        s = Settings()
        assert s.session_history_limit == 20

    def test_max_context_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "8192")
        s = Settings()
        assert s.max_context_tokens == 8192

    def test_rate_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "120")
        monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "50")
        s = Settings()
        assert s.rate_limit_window_seconds == 120
        assert s.rate_limit_max_requests == 50


class TestValidation:
    """Verify Pydantic validation rejects invalid values."""

    def test_session_history_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(session_history_limit=0)

    def test_rate_limit_window_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(rate_limit_window_seconds=0)

    def test_rate_limit_max_requests_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(rate_limit_max_requests=0)

    def test_max_context_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(max_context_tokens=0)

    def test_retry_max_attempts_allows_zero(self):
        s = Settings(retry_max_attempts=0)
        assert s.retry_max_attempts == 0


class TestGetSettings:
    """Verify the factory function."""

    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)
