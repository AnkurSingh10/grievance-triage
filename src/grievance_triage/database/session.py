from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.settings import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from . import models  # noqa: F401

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS district VARCHAR(150)"))
        connection.execute(text("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS source_dataset_hash VARCHAR(64)"))
        connection.execute(text("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS source_split VARCHAR(20)"))
        connection.execute(text("ALTER TABLE complaints ADD COLUMN IF NOT EXISTS source_row_number INTEGER"))
        connection.execute(text("ALTER TABLE issue_groups DROP COLUMN IF EXISTS district"))
