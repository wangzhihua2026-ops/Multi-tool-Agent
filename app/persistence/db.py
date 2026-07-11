from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_database(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(normalize_async_database_url(database_url), pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def normalize_async_database_url(database_url: str) -> str:
    """Accept managed Postgres URLs while always selecting the asyncpg driver."""
    if database_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + database_url.removeprefix("postgresql://")
    return database_url
