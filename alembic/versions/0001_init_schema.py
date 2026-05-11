"""Initial schema with pgvector extension."""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

from mega_ai.db.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
