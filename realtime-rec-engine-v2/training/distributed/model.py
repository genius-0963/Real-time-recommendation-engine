"""
Distributed recommendation model using PyTorch DDP.
Implements a two-tower architecture with user and item embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature interaction."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project to multi-head
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        
        return self.out_proj(context)


class FeatureTransformer(nn.Module):
    """Transformer encoder for feature processing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.feature_encoder = nn.Sequential(*layers)
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode features
        encoded = self.feature_encoder(features)
        
        # Add sequence dimension for attention
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)
        
        # Self-attention
        attended = self.attention(encoded, encoded, encoded, mask)
        
        # Residual connection and normalization
        output = self.layer_norm(encoded + self.dropout(attended))
        
        # Remove sequence dimension if added
        if output.shape[1] == 1:
            output = output.squeeze(1)
        
        return output


class UserTower(nn.Module):
    """User tower of the two-tower model."""
    
    def __init__(self, 
                 user_vocab_size: int,
                 item_vocab_size: int,
                 embedding_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.2,
                 num_heads: int = 8):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_history_embedding = nn.Embedding(item_vocab_size, embedding_dim)
        
        # Feature encoders
        self.user_feature_encoder = FeatureTransformer(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.history_encoder = FeatureTransformer(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], embedding_dim)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_history_embedding.weight)
    
    def forward(self, 
                user_ids: torch.Tensor,
                user_features: torch.Tensor,
                item_history: torch.Tensor,
                history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # User embedding
        user_emb = self.user_embedding(user_ids)
        user_emb = self.user_feature_encoder(user_emb)
        
        # Item history embedding
        history_emb = self.item_history_embedding(item_history)
        history_emb = self.history_encoder(history_emb, history_mask)
        
        # Aggregate history (mean pooling)
        if history_mask is not None:
            history_emb = history_emb * history_mask.unsqueeze(-1)
            history_lengths = history_mask.sum(dim=1, keepdim=True)
            history_emb = history_emb.sum(dim=1) / (history_lengths + 1e-8)
        else:
            history_emb = history_emb.mean(dim=1)
        
        # Fuse user and history representations
        combined = torch.cat([user_emb, history_emb], dim=-1)
        user_representation = self.fusion_layers(combined)
        
        return F.normalize(user_representation, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower of the two-tower model."""
    
    def __init__(self,
                 item_vocab_size: int,
                 category_vocab_size: int,
                 embedding_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.2,
                 num_heads: int = 8):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(category_vocab_size, embedding_dim // 4)
        
        # Feature encoders
        self.item_feature_encoder = FeatureTransformer(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion layers
        total_dim = embedding_dim + (embedding_dim // 4)  # item + category
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], embedding_dim)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
    
    def forward(self,
                item_ids: torch.Tensor,
                item_features: torch.Tensor,
                category_ids: torch.Tensor) -> torch.Tensor:
        
        # Item embedding
        item_emb = self.item_embedding(item_ids)
        item_emb = self.item_feature_encoder(item_emb)
        
        # Category embedding
        category_emb = self.category_embedding(category_ids)
        
        # Fuse item and category representations
        combined = torch.cat([item_emb, category_emb], dim=-1)
        item_representation = self.fusion_layers(combined)
        
        return F.normalize(item_representation, p=2, dim=-1)


class TwoTowerRecommendationModel(nn.Module):
    """Two-tower recommendation model for distributed training."""
    
    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 category_vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.2,
                 num_heads: int = 8,
                 temperature: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        self.user_tower = UserTower(
            user_vocab_size=user_vocab_size,
            item_vocab_size=item_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_heads=num_heads
        )
        
        self.item_tower = ItemTower(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_heads=num_heads
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, 
                user_ids: torch.Tensor,
                user_features: torch.Tensor,
                item_history: torch.Tensor,
                history_mask: torch.Tensor,
                item_ids: torch.Tensor,
                item_features: torch.Tensor,
                category_ids: torch.Tensor,
                negative_item_ids: Optional[torch.Tensor] = None,
                negative_item_features: Optional[torch.Tensor] = None,
                negative_category_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = user_ids.shape[0]
        
        # Get user representations
        user_repr = self.user_tower(
            user_ids=user_ids,
            user_features=user_features,
            item_history=item_history,
            history_mask=history_mask
        )
        
        # Get positive item representations
        item_repr = self.item_tower(
            item_ids=item_ids,
            item_features=item_features,
            category_ids=category_ids
        )
        
        # Compute positive scores
        positive_scores = torch.sum(user_repr * item_repr, dim=-1) / self.temperature
        
        if negative_item_ids is not None:
            # Get negative item representations
            neg_item_repr = self.item_tower(
                item_ids=negative_item_ids,
                item_features=negative_item_features,
                category_ids=negative_category_ids
            )
            
            # Compute negative scores
            # Reshape for broadcasting: [batch_size, num_negatives, embedding_dim]
            user_repr_expanded = user_repr.unsqueeze(1)
            negative_scores = torch.sum(user_repr_expanded * neg_item_repr, dim=-1) / self.temperature
            
            # Combine positive and negative scores
            # Shape: [batch_size, 1 + num_negatives]
            all_scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
            
            # Labels: 0 for positive item
            labels = torch.zeros(batch_size, dtype=torch.long, device=user_ids.device)
            
            # Compute loss
            loss = self.criterion(all_scores, labels)
            
            # Compute metrics
            with torch.no_grad():
                # Accuracy@1
                predictions = all_scores.argmax(dim=-1)
                accuracy = (predictions == 0).float().mean()
                
                # Recall@K
                k_values = [1, 5, 10]
                recalls = {}
                for k in k_values:
                    if all_scores.shape[1] >= k:
                        _, top_k_indices = all_scores.topk(k, dim=-1)
                        recall_at_k = (top_k_indices == 0).any(dim=-1).float().mean()
                        recalls[f'recall@{k}'] = recall_at_k
            
            return {
                'loss': loss,
                'positive_scores': positive_scores,
                'negative_scores': negative_scores,
                'accuracy': accuracy,
                **recalls
            }
        
        else:
            # Inference mode - only positive scores
            return {
                'user_representation': user_repr,
                'item_representation': item_repr,
                'positive_scores': positive_scores
            }
    
    def get_user_embeddings(self, 
                           user_ids: torch.Tensor,
                           user_features: torch.Tensor,
                           item_history: torch.Tensor,
                           history_mask: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for inference."""
        return self.user_tower(
            user_ids=user_ids,
            user_features=user_features,
            item_history=item_history,
            history_mask=history_mask
        )
    
    def get_item_embeddings(self,
                           item_ids: torch.Tensor,
                           item_features: torch.Tensor,
                           category_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings for inference."""
        return self.item_tower(
            item_ids=item_ids,
            item_features=item_features,
            category_ids=category_ids
        )
    
    def compute_similarity_matrix(self, 
                                 user_embeddings: torch.Tensor,
                                 item_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between users and items."""
        return torch.matmul(user_embeddings, item_embeddings.T)


class DistributedModelWrapper:
    """Wrapper for distributed training with DDP."""
    
    def __init__(self, model: nn.Module, local_rank: int):
        self.model = model
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if multiple GPUs
        if torch.cuda.device_count() > 1:
            import torch.distributed as dist
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
                  scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(**batch)
            loss = outputs['loss']
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Return metrics
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in outputs.items()}
        return metrics
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
            
            metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in outputs.items()}
            return metrics
