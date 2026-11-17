"""No live desktop or heavyweight model is needed for geometry regression tests."""
import pytest


@pytest.fixture(autouse=True)
def default_fixture_ocr(monkeypatch):
    from welog import controller
    from welog.recognition import EmptyOcr
    monkeypatch.setattr(controller,"RapidOcrAdapter",EmptyOcr)
