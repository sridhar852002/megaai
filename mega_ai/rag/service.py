"""Multi-hop vector retrieval."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from litellm import aembedding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.hashing import sha256_text
from mega_ai.core.settings import Settings, get_settings
from mega_ai.db import models


def _mock_embedding(content: str, dim: int = 1536) -> list[float]:
    seed = sha256_text(content)
    rng = np.random.default_rng(int(seed[:8], 16))
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= float(np.linalg.norm(vec) + 1e-9)
    return [float(x) for x in vec.tolist()]


async def embed_text(settings: Settings, content: str) -> list[float]:
    if settings.llm_mode.lower() == "mock":
        return _mock_embedding(content)
    response = await aembedding(model=settings.rag_embedding_model, input=content)
    return list(response["data"][0]["embedding"])


async def multi_hop_retrieve(
    session: AsyncSession,
    *,
    hop1_query: str,
    hop2_query: str,
    settings: Settings | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    e1 = await embed_text(cfg, hop1_query)
    e2 = await embed_text(cfg, hop2_query)

    stmt1 = select(models.ChunkRow).order_by(models.ChunkRow.embedding.cosine_distance(e1)).limit(top_k)
    stmt2 = select(models.ChunkRow).order_by(models.ChunkRow.embedding.cosine_distance(e2)).limit(top_k)
    chunks_round1 = (await session.execute(stmt1)).scalars().all()
    chunks_round2 = (await session.execute(stmt2)).scalars().all()

    merged: dict[uuid.UUID, models.ChunkRow] = {}
    for c in list(chunks_round1) + list(chunks_round2):
        merged[c.id] = c

    ordered = list(merged.values())
    return {
        "chunks": [
            {
                "chunk_id": str(c.id),
                "document_id": str(c.document_id),
                "text": c.text,
                "metadata": c.chunk_metadata,
            }
            for c in ordered
        ],
        "hop1_query": hop1_query,
        "hop2_query": hop2_query,
    }


async def seed_demo_corpus(session: AsyncSession, settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    exists = await session.execute(select(models.ChunkRow.id).limit(1))
    if exists.scalar_one_or_none():
        return

    doc = models.DocumentRow(title="World Factbook (demo)", source_uri="internal://demo")
    session.add(doc)
    await session.flush()

    snippets = [
        (
            "France is a country in Western Europe. Its capital is Paris, located on the Seine.",
            {"topic": "france_capital", "hop": "geo"},
        ),
        (
            "Paris hosts the Louvre and is a global center for art and diplomacy.",
            {"topic": "paris_context", "hop": "culture"},
        ),
        (
            "The European Union has institutions in Brussels, Strasbourg, and Luxembourg.",
            {"topic": "eu_institutions", "hop": "politics"},
        ),
    ]
    for text, meta in snippets:
        emb = await embed_text(cfg, text)
        session.add(
            models.ChunkRow(
                document_id=doc.id,
                text=text,
                embedding=emb,
                chunk_metadata=meta,
            )
        )
