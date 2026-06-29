import pytest


@pytest.fixture(autouse=True)
def set_fake_api_key(monkeypatch):
    """Set a dummy API key for all tests so os.environ lookups don't raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-tests")
