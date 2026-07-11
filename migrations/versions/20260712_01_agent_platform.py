"""Create durable Agent platform tables.

Revision ID: 20260712_01
Revises:
Create Date: 2026-07-12
"""

from alembic import op

from app.persistence.platform_tables import metadata


revision = "20260712_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    metadata.drop_all(bind=op.get_bind())
