"""Initial schema - features, interactions, model_versions tables

Revision ID: 001
Revises: None
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Features table ────────────────────────────
    op.create_table(
        "features",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.VARCHAR(50), nullable=False),
        sa.Column("entity_id", sa.VARCHAR(100), nullable=False),
        sa.Column("feature_name", sa.VARCHAR(100), nullable=False),
        sa.Column("feature_value", JSONB(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_features_entity_type_entity_id",
        "features",
        ["entity_type", "entity_id"],
    )
    op.create_index(
        "ix_features_entity_type_entity_id_feature_name",
        "features",
        ["entity_type", "entity_id", "feature_name"],
    )

    # ── Interactions table ────────────────────────
    op.create_table(
        "interactions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.VARCHAR(100), nullable=False),
        sa.Column("item_id", sa.VARCHAR(100), nullable=False),
        sa.Column("event_type", sa.VARCHAR(50), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("context", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_interactions_user_id", "interactions", ["user_id"])
    op.create_index("ix_interactions_item_id", "interactions", ["item_id"])
    op.create_index("ix_interactions_timestamp", "interactions", ["timestamp"])

    # ── Model Versions table ─────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.VARCHAR(50), nullable=False, unique=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column(
            "status",
            sa.VARCHAR(20),
            server_default="active",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", name="uq_model_versions_version"),
    )
    op.create_index("ix_model_versions_version", "model_versions", ["version"])
    op.create_index("ix_model_versions_status", "model_versions", ["status"])


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("interactions")
    op.drop_table("features")
