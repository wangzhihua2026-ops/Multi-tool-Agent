from app.core.config import Settings


def test_document_ocr_settings_have_safe_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DOCUMENT_OCR_ENABLED", raising=False)
    monkeypatch.delenv("DOCUMENT_OCR_MAX_PAGES", raising=False)
    monkeypatch.delenv("DOCUMENT_OCR_MIN_NATIVE_CHARS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.document_ocr_enabled is True
    assert settings.document_ocr_max_pages == 50
    assert settings.document_ocr_min_native_chars == 50


def test_document_ocr_settings_can_be_configured(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_ENABLED", "false")
    monkeypatch.setenv("DOCUMENT_OCR_MAX_PAGES", "7")
    monkeypatch.setenv("DOCUMENT_OCR_MIN_NATIVE_CHARS", "120")

    settings = Settings(_env_file=None)

    assert settings.document_ocr_enabled is False
    assert settings.document_ocr_max_pages == 7
    assert settings.document_ocr_min_native_chars == 120
