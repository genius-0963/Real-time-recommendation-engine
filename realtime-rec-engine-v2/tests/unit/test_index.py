"""
Unit tests for ANN index building (FAISS-based).

Source: index/build_index.py
Classes tested:
  - IndexConfig
  - FAISSIndex
  - IndexManager
"""

import os
import tempfile
import pytest
import numpy as np

# FAISS may or may not be installed — skip the entire module if not.
faiss = pytest.importorskip("faiss", reason="FAISS not installed")

from index.build_index import (
    IndexConfig,
    FAISSIndex,
    IndexManager,
    BaseANNIndex,
    FAISS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_faiss_config(**overrides) -> IndexConfig:
    defaults = dict(
        index_type="faiss",
        embedding_dim=128,
        num_items=100,
        index_path=tempfile.mkdtemp(),
        faiss_index_type="HNSW",   # HNSW doesn't need training → simpler for tests
        faiss_nlist=10,
        faiss_m=32,
        faiss_nbits=8,
    )
    defaults.update(overrides)
    return IndexConfig(**defaults)


# ===================================================================
# Tests
# ===================================================================


class TestFaissIndexBuild:
    """Build a FAISS index and verify is_built flag."""

    def test_faiss_index_build(self, sample_embeddings, sample_item_ids):
        config = _make_faiss_config()
        index = FAISSIndex(config)
        result = index.build(sample_embeddings, sample_item_ids)

        assert result is True
        assert index.is_built is True
        assert index.index is not None


class TestFaissIndexSearch:
    """Search returns k results."""

    def test_faiss_index_search(self, sample_embeddings, sample_item_ids):
        config = _make_faiss_config()
        index = FAISSIndex(config)
        index.build(sample_embeddings, sample_item_ids)

        query = np.random.RandomState(0).randn(128).astype(np.float32)
        k = 5
        item_ids, scores = index.search(query, k=k)

        assert len(item_ids) == k
        assert len(scores) == k
        # All returned IDs should be from the original list
        assert all(iid in sample_item_ids for iid in item_ids)


class TestFaissIndexSaveLoad:
    """Save, reload, and search still works."""

    def test_faiss_index_save_load(self, sample_embeddings, sample_item_ids):
        config = _make_faiss_config()
        index = FAISSIndex(config)
        index.build(sample_embeddings, sample_item_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            assert index.save(tmpdir) is True

            # Load into a fresh instance
            loaded_index = FAISSIndex(config)
            assert loaded_index.load(tmpdir) is True
            assert loaded_index.is_built is True

            # Search on the loaded index
            query = np.random.RandomState(1).randn(128).astype(np.float32)
            item_ids, scores = loaded_index.search(query, k=5)
            assert len(item_ids) == 5


class TestIndexManagerBuild:
    """IndexManager.build_index works end-to-end."""

    def test_index_manager_build(self, sample_embeddings, sample_item_ids):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_faiss_config(index_path=tmpdir)
            manager = IndexManager(config)
            result = manager.build_index(sample_embeddings, sample_item_ids)

            assert result is True
            assert manager.active_index is not None
            assert manager.active_index.is_built is True


class TestIndexManagerSearch:
    """IndexManager.search returns results after building."""

    def test_index_manager_search(self, sample_embeddings, sample_item_ids):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_faiss_config(index_path=tmpdir)
            manager = IndexManager(config)
            manager.build_index(sample_embeddings, sample_item_ids)

            query = np.random.RandomState(2).randn(128).astype(np.float32)
            item_ids, scores = manager.search(query, k=5)

            assert len(item_ids) == 5
            assert len(scores) == 5


class TestIndexConfigDefaults:
    """IndexConfig has correct defaults."""

    def test_index_config_defaults(self):
        config = IndexConfig()
        assert config.index_type == "scann"
        assert config.embedding_dim == 128
        assert config.num_items == 1_000_000
        assert config.faiss_index_type == "IVF_PQ"
        assert config.faiss_nlist == 1000
        assert config.faiss_m == 64
        assert config.faiss_nbits == 8
        assert config.batch_size == 10000
