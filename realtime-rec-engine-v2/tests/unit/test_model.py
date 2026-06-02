"""
Unit tests for the Two-Tower recommendation model.

Source: training/distributed/model.py
Classes tested:
  - TwoTowerRecommendationModel
  - UserTower
  - ItemTower
  - MultiHeadAttention
  - FeatureTransformer
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from training.distributed.model import (
    TwoTowerRecommendationModel,
    UserTower,
    ItemTower,
    MultiHeadAttention,
    FeatureTransformer,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
USER_VOCAB = 1000
ITEM_VOCAB = 5000
CATEGORY_VOCAB = 50
EMBEDDING_DIM = 128
HIDDEN_DIMS = [512, 256, 128]
BATCH_SIZE = 8
HISTORY_LEN = 10


# ---------------------------------------------------------------------------
# Helper to create a model with default test sizes
# ---------------------------------------------------------------------------
def _make_model(**overrides):
    defaults = dict(
        user_vocab_size=USER_VOCAB,
        item_vocab_size=ITEM_VOCAB,
        category_vocab_size=CATEGORY_VOCAB,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
    )
    defaults.update(overrides)
    return TwoTowerRecommendationModel(**defaults)


def _sample_batch(batch_size=BATCH_SIZE, with_negatives=False, num_negatives=5):
    """Return a dict of tensors that can be unpacked into model.forward()."""
    batch = dict(
        user_ids=torch.randint(0, USER_VOCAB, (batch_size,)),
        user_features=torch.randn(batch_size, EMBEDDING_DIM),
        item_history=torch.randint(0, ITEM_VOCAB, (batch_size, HISTORY_LEN)),
        history_mask=torch.ones(batch_size, HISTORY_LEN),
        item_ids=torch.randint(0, ITEM_VOCAB, (batch_size,)),
        item_features=torch.randn(batch_size, EMBEDDING_DIM),
        category_ids=torch.randint(0, CATEGORY_VOCAB, (batch_size,)),
    )
    if with_negatives:
        batch["negative_item_ids"] = torch.randint(0, ITEM_VOCAB, (batch_size, num_negatives))
        batch["negative_item_features"] = torch.randn(batch_size, num_negatives, EMBEDDING_DIM)
        batch["negative_category_ids"] = torch.randint(0, CATEGORY_VOCAB, (batch_size, num_negatives))
    return batch


# ===================================================================
# Tests
# ===================================================================


class TestModelCreation:
    """Test that the model can be instantiated correctly."""

    def test_model_creation(self):
        """Instantiate TwoTowerRecommendationModel and check it is an nn.Module."""
        model = _make_model()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "user_tower")
        assert hasattr(model, "item_tower")
        assert model.embedding_dim == EMBEDDING_DIM

    def test_user_tower_is_module(self):
        model = _make_model()
        assert isinstance(model.user_tower, nn.Module)

    def test_item_tower_is_module(self):
        model = _make_model()
        assert isinstance(model.item_tower, nn.Module)


class TestForwardPass:
    """Test forward pass with and without negatives."""

    def test_forward_pass(self):
        """Forward pass without negatives returns user_repr, item_repr, positive_scores."""
        model = _make_model()
        model.eval()
        batch = _sample_batch()

        with torch.no_grad():
            outputs = model(**batch)

        assert "positive_scores" in outputs
        assert outputs["positive_scores"].shape == (BATCH_SIZE,)
        assert "user_representation" in outputs
        assert "item_representation" in outputs

    def test_forward_pass_with_negatives(self):
        """Forward pass with negatives returns loss, accuracy, etc."""
        model = _make_model()
        model.eval()
        num_negatives = 5
        batch = _sample_batch(with_negatives=True, num_negatives=num_negatives)

        with torch.no_grad():
            outputs = model(**batch)

        assert "loss" in outputs
        assert "accuracy" in outputs
        assert "positive_scores" in outputs
        assert "negative_scores" in outputs
        assert outputs["negative_scores"].shape == (BATCH_SIZE, num_negatives)


class TestEmbeddingGeneration:
    """Test dedicated embedding extraction methods."""

    def test_embedding_generation(self):
        """get_user_embeddings / get_item_embeddings return (batch, 128)."""
        model = _make_model()
        model.eval()
        batch = _sample_batch()

        with torch.no_grad():
            user_emb = model.get_user_embeddings(
                user_ids=batch["user_ids"],
                user_features=batch["user_features"],
                item_history=batch["item_history"],
                history_mask=batch["history_mask"],
            )
            item_emb = model.get_item_embeddings(
                item_ids=batch["item_ids"],
                item_features=batch["item_features"],
                category_ids=batch["category_ids"],
            )

        assert user_emb.shape == (BATCH_SIZE, EMBEDDING_DIM)
        assert item_emb.shape == (BATCH_SIZE, EMBEDDING_DIM)


class TestPredict:
    """Test prediction (scoring) via the similarity matrix helper."""

    def test_predict(self):
        """compute_similarity_matrix returns valid scores."""
        model = _make_model()
        model.eval()
        batch = _sample_batch()

        with torch.no_grad():
            user_emb = model.get_user_embeddings(
                user_ids=batch["user_ids"],
                user_features=batch["user_features"],
                item_history=batch["item_history"],
                history_mask=batch["history_mask"],
            )
            item_emb = model.get_item_embeddings(
                item_ids=batch["item_ids"],
                item_features=batch["item_features"],
                category_ids=batch["category_ids"],
            )
            similarity = model.compute_similarity_matrix(user_emb, item_emb)

        assert similarity.shape == (BATCH_SIZE, BATCH_SIZE)
        # Cosine similarity values should be roughly in [-1, 1]
        assert similarity.max().item() <= 1.01
        assert similarity.min().item() >= -1.01


class TestLossComputation:
    """Test that the loss is a scalar tensor when negatives are provided."""

    def test_loss_computation(self):
        """Loss should be a zero-dim (scalar) tensor."""
        model = _make_model()
        model.train()
        batch = _sample_batch(with_negatives=True)

        outputs = model(**batch)
        loss = outputs["loss"]

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.requires_grad


class TestModelSaveLoad:
    """Test round-trip save / load of model weights."""

    def test_model_save_load(self):
        """Save model to a temp dir, load it back, verify parameters match."""
        model_a = _make_model()
        model_b = _make_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            torch.save(model_a.state_dict(), save_path)
            model_b.load_state_dict(torch.load(save_path, weights_only=True))

        # Verify every parameter matches
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            assert name_a == name_b
            assert torch.equal(param_a, param_b), f"Mismatch in parameter {name_a}"
