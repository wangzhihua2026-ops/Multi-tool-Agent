from app.persistence.db import normalize_async_database_url


def test_managed_postgres_urls_select_asyncpg_driver() -> None:
    assert normalize_async_database_url("postgres://u:p@host/db") == "postgresql+asyncpg://u:p@host/db"
    assert normalize_async_database_url("postgresql://u:p@host/db") == "postgresql+asyncpg://u:p@host/db"


def test_explicit_async_driver_is_preserved() -> None:
    value = "postgresql+asyncpg://u:p@host/db"
    assert normalize_async_database_url(value) == value
