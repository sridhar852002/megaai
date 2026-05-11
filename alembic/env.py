import os

from alembic import context
from sqlalchemy import create_engine

from mega_ai.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = os.environ["DATABASE_SYNC_URL"]
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(os.environ["DATABASE_SYNC_URL"])

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
