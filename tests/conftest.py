import pytest

from steerquant.smoke import fixture_rows


@pytest.fixture
def raw_rows():
    return fixture_rows()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
