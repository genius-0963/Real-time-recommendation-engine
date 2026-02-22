"""
Dataset classes for distributed training of recommendation models.
Supports sharded loading, negative sampling, and efficient data pipelines.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle
import h5py
from pathlib import Path
import random
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class InteractionDataset(Dataset):
    """Dataset for user-item interactions with negative sampling."""
    
    def __init__(self,
                 interactions_df: pd.DataFrame,
                 user_features_df: pd.DataFrame,
                 item_features_df: pd.DataFrame,
                 item_popularity: Optional[Dict[int, int]] = None,
                 num_negatives: int = 5,
                 negative_sampling_strategy: str = "uniform",
                 max_history_length: int = 50,
                 history_padding_value: int = 0,
                 transform: Optional[callable] = None):
        
        self.interactions = interactions_df
        self.user_features = user_features_df
        self.item_features = item_features_df
        self.num_negatives = num_negatives
        self.negative_sampling_strategy = negative_sampling_strategy
        self.max_history_length = max_history_length
        self.history_padding_value = history_padding_value
        self.transform = transform
        
        # Create mappings
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_features_df['user_id'].unique())}
        self.item_id_map = {iid: idx for idx, iid in enumerate(item_features_df['item_id'].unique())}
        
        # Reverse mappings for negative sampling
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_map.items()}
        self.num_items = len(self.item_id_map)
        
        # Build user interaction history
        self._build_user_history()
        
        # Prepare negative sampling
        self._prepare_negative_sampling(item_popularity)
        
        logger.info(f"Dataset initialized with {len(self.interactions)} interactions")
        logger.info(f"Users: {len(self.user_id_map)}, Items: {len(self.item_id_map)}")
    
    def _build_user_history(self):
        """Build user interaction history for sequence modeling."""
        self.user_history = defaultdict(list)
        
        # Sort interactions by timestamp for each user
        sorted_interactions = self.interactions.sort_values(['user_id', 'timestamp'])
        
        for _, row in sorted_interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            
            if user_id in self.user_id_map and item_id in self.item_id_map:
                self.user_history[user_id].append(item_id)
        
        # Convert to mapped IDs
        self.mapped_user_history = {}
        for user_id, items in self.user_history.items():
            mapped_items = [self.item_id_map[iid] for iid in items if iid in self.item_id_map]
            self.mapped_user_history[self.user_id_map[user_id]] = mapped_items
    
    def _prepare_negative_sampling(self, item_popularity: Optional[Dict[int, int]]):
        """Prepare negative sampling strategy."""
        if item_popularity is None:
            # Count item frequencies from interactions
            item_counts = Counter(self.interactions['item_id'])
            item_popularity = {iid: count for iid, count in item_counts.items() if iid in self.item_id_map}
        
        if self.negative_sampling_strategy == "uniform":
            # Uniform sampling
            self.negative_probs = np.ones(self.num_items) / self.num_items
        
        elif self.negative_sampling_strategy == "popular":
            # Sample based on popularity
            probs = np.zeros(self.num_items)
            for item_id, count in item_popularity.items():
                if item_id in self.item_id_map:
                    probs[self.item_id_map[item_id]] = count
            self.negative_probs = probs / probs.sum()
        
        elif self.negative_sampling_strategy == "adaptive":
            # Inverse popularity sampling (less popular items get higher probability)
            probs = np.zeros(self.num_items)
            for item_id, count in item_popularity.items():
                if item_id in self.item_id_map:
                    probs[self.item_id_map[item_id]] = 1.0 / (count + 1)
            self.negative_probs = probs / probs.sum()
        
        else:
            raise ValueError(f"Unknown negative sampling strategy: {self.negative_sampling_strategy}")
    
    def _sample_negatives(self, user_id: int, positive_item_id: int, num_negatives: int) -> List[int]:
        """Sample negative items for a user."""
        negatives = []
        user_history_set = set(self.mapped_user_history.get(user_id, []))
        positive_mapped = self.item_id_map.get(positive_item_id, -1)
        
        while len(negatives) < num_negatives:
            # Sample based on strategy
            if self.negative_sampling_strategy == "uniform":
                neg_idx = random.randint(0, self.num_items - 1)
            else:
                neg_idx = np.random.choice(self.num_items, p=self.negative_probs)
            
            # Ensure it's not in user history and not the positive item
            if (neg_idx not in user_history_set and 
                neg_idx != positive_mapped and 
                neg_idx not in negatives):
                negatives.append(neg_idx)
        
        return negatives
    
    def _get_user_history_sequence(self, user_id: int, current_item_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user interaction history sequence and mask."""
        user_mapped_id = self.user_id_map[user_id]
        history = self.mapped_user_history.get(user_mapped_id, [])
        
        # Remove current item from history if present
        current_mapped = self.item_id_map.get(current_item_id, -1)
        if current_mapped in history:
            history = [item for item in history if item != current_mapped]
        
        # Take last N interactions
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        
        # Convert to tensor
        history_tensor = torch.tensor(history, dtype=torch.long)
        mask = torch.ones(len(history), dtype=torch.float)
        
        return history_tensor, mask
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        item_id = row['item_id']
        
        # Skip if user or item not in mappings
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            # Return a dummy sample
            return self._get_dummy_sample()
        
        # Get mapped IDs
        user_mapped_id = self.user_id_map[user_id]
        item_mapped_id = self.item_id_map[item_id]
        
        # Get user features
        user_features = self.user_features[self.user_features['user_id'] == user_id].iloc[0]
        user_features_tensor = torch.tensor(user_features.drop('user_id').values, dtype=torch.float)
        
        # Get item features
        item_features = self.item_features[self.item_features['item_id'] == item_id].iloc[0]
        item_features_tensor = torch.tensor(item_features.drop(['item_id', 'category_id']).values, dtype=torch.float)
        category_id = item_features['category_id']
        
        # Get user history
        history_tensor, history_mask = self._get_user_history_sequence(user_id, item_id)
        
        # Sample negative items
        negative_item_ids = self._sample_negatives(user_mapped_id, item_id, self.num_negatives)
        negative_item_features = []
        negative_category_ids = []
        
        for neg_idx in negative_item_ids:
            neg_original_id = self.idx_to_item_id[neg_idx]
            neg_features = self.item_features[self.item_features['item_id'] == neg_original_id].iloc[0]
            negative_item_features.append(neg_features.drop(['item_id', 'category_id']).values)
            negative_category_ids.append(neg_features['category_id'])
        
        negative_item_features_tensor = torch.tensor(negative_item_features, dtype=torch.float)
        negative_category_ids_tensor = torch.tensor(negative_category_ids, dtype=torch.long)
        
        sample = {
            'user_ids': torch.tensor(user_mapped_id, dtype=torch.long),
            'user_features': user_features_tensor,
            'item_history': history_tensor,
            'history_mask': history_mask,
            'item_ids': torch.tensor(item_mapped_id, dtype=torch.long),
            'item_features': item_features_tensor,
            'category_ids': torch.tensor(category_id, dtype=torch.long),
            'negative_item_ids': torch.tensor(negative_item_ids, dtype=torch.long),
            'negative_item_features': negative_item_features_tensor,
            'negative_category_ids': negative_category_ids_tensor
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for invalid interactions."""
        return {
            'user_ids': torch.tensor(0, dtype=torch.long),
            'user_features': torch.zeros(10, dtype=torch.float),
            'item_history': torch.zeros(1, dtype=torch.long),
            'history_mask': torch.zeros(1, dtype=torch.float),
            'item_ids': torch.tensor(0, dtype=torch.long),
            'item_features': torch.zeros(10, dtype=torch.float),
            'category_ids': torch.tensor(0, dtype=torch.long),
            'negative_item_ids': torch.zeros(self.num_negatives, dtype=torch.long),
            'negative_item_features': torch.zeros(self.num_negatives, 10, dtype=torch.float),
            'negative_category_ids': torch.zeros(self.num_negatives, dtype=torch.long)
        }


class HDF5InteractionDataset(Dataset):
    """Memory-efficient dataset using HDF5 storage for large-scale training."""
    
    def __init__(self,
                 hdf5_path: str,
                 mode: str = 'train',
                 num_negatives: int = 5,
                 max_history_length: int = 50):
        
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.num_negatives = num_negatives
        self.max_history_length = max_history_length
        
        # Open HDF5 file
        self.h5_file = h5py.File(hdf5_path, 'r')
        self.dataset = self.h5_file[mode]
        
        # Load metadata
        self.metadata = dict(self.h5_file['metadata'])
        self.num_items = int(self.metadata['num_items'])
        
        logger.info(f"Loaded {mode} dataset with {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from HDF5 storage."""
        sample = self.dataset[str(idx)]
        
        # Convert to tensors
        result = {
            'user_ids': torch.tensor(sample['user_id'][()], dtype=torch.long),
            'user_features': torch.tensor(sample['user_features'][()], dtype=torch.float),
            'item_history': torch.tensor(sample['item_history'][()], dtype=torch.long),
            'history_mask': torch.tensor(sample['history_mask'][()], dtype=torch.float),
            'item_ids': torch.tensor(sample['item_id'][()], dtype=torch.long),
            'item_features': torch.tensor(sample['item_features'][()], dtype=torch.float),
            'category_ids': torch.tensor(sample['category_id'][()], dtype=torch.long),
            'negative_item_ids': torch.tensor(sample['negative_item_ids'][()], dtype=torch.long),
            'negative_item_features': torch.tensor(sample['negative_item_features'][()], dtype=torch.float),
            'negative_category_ids': torch.tensor(sample['negative_category_ids'][()], dtype=torch.long)
        }
        
        return result
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    batch_size = len(batch)
    
    # Handle fixed-size tensors
    result = {
        'user_ids': torch.stack([item['user_ids'] for item in batch]),
        'user_features': torch.stack([item['user_features'] for item in batch]),
        'item_ids': torch.stack([item['item_ids'] for item in batch]),
        'item_features': torch.stack([item['item_features'] for item in batch]),
        'category_ids': torch.stack([item['category_ids'] for item in batch])
    }
    
    # Handle variable-length sequences with padding
    # Item history
    history_lengths = [len(item['item_history']) for item in batch]
    max_history_len = max(history_lengths)
    padded_history = pad_sequence(
        [item['item_history'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    result['item_history'] = padded_history
    
    # History mask
    padded_mask = pad_sequence(
        [item['history_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    result['history_mask'] = padded_mask
    
    # Handle negative samples (batch_size x num_negatives x feature_dim)
    num_negatives = batch[0]['negative_item_ids'].shape[0]
    result['negative_item_ids'] = torch.stack([item['negative_item_ids'] for item in batch])
    result['negative_item_features'] = torch.stack([item['negative_item_features'] for item in batch])
    result['negative_category_ids'] = torch.stack([item['negative_category_ids'] for item in batch])
    
    return result


def create_distributed_dataloader(dataset: Dataset,
                                batch_size: int,
                                num_workers: int = 4,
                                pin_memory: bool = True,
                                persistent_workers: bool = True,
                                shuffle: bool = True) -> DataLoader:
    """Create a distributed data loader for multi-GPU training."""
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return dataloader


class DataPreprocessor:
    """Preprocessor for preparing training data."""
    
    def __init__(self, min_interactions: int = 5, min_item_interactions: int = 5):
        self.min_interactions = min_interactions
        self.min_item_interactions = min_item_interactions
    
    def preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess interaction data by filtering and cleaning."""
        logger.info(f"Original interactions: {len(interactions_df)}")
        
        # Remove duplicates
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])
        
        # Filter users with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = interactions_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(valid_items)]
        
        logger.info(f"Filtered interactions: {len(interactions_df)}")
        logger.info(f"Users: {interactions_df['user_id'].nunique()}, Items: {interactions_df['item_id'].nunique()}")
        
        return interactions_df
    
    def create_hdf5_dataset(self,
                          interactions_df: pd.DataFrame,
                          user_features_df: pd.DataFrame,
                          item_features_df: pd.DataFrame,
                          output_path: str,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> None:
        """Create HDF5 dataset for efficient loading."""
        
        # Shuffle interactions
        interactions_df = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split data
        n = len(interactions_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = interactions_df[:train_end]
        val_df = interactions_df[train_end:val_end]
        test_df = interactions_df[val_end:]
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as h5_file:
            # Store metadata
            metadata = h5_file.create_group('metadata')
            metadata.attrs['num_items'] = len(item_features_df)
            metadata.attrs['num_users'] = len(user_features_df)
            metadata.attrs['train_size'] = len(train_df)
            metadata.attrs['val_size'] = len(val_df)
            metadata.attrs['test_size'] = len(test_df)
            
            # Process each split
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                split_group = h5_file.create_group(split_name)
                
                # Create dataset for this split
                dataset = InteractionDataset(
                    interactions_df=split_df,
                    user_features_df=user_features_df,
                    item_features_df=item_features_df,
                    num_negatives=5
                )
                
                # Store samples
                for i in range(len(dataset)):
                    sample = dataset[i]
                    sample_group = split_group.create_group(str(i))
                    
                    for key, value in sample.items():
                        sample_group.create_dataset(key, data=value.numpy())
        
        logger.info(f"HDF5 dataset created at {output_path}")


# Utility functions for data loading
def load_sample_data(num_users: int = 10000, num_items: int = 50000, num_interactions: int = 1000000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sample data for testing."""
    
    # Generate interactions
    np.random.seed(42)
    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    timestamps = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_interactions), unit='D')
    ratings = np.random.randint(1, 6, num_interactions)
    
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'timestamp': timestamps,
        'rating': ratings
    })
    
    # Generate user features
    user_features = []
    for user_id in range(1, num_users + 1):
        features = {
            'user_id': user_id,
            'age': np.random.randint(18, 65),
            'gender': np.random.choice(['M', 'F']),
            'income': np.random.randint(30000, 150000),
            'membership_days': np.random.randint(1, 1000)
        }
        user_features.append(features)
    
    user_features_df = pd.DataFrame(user_features)
    
    # Generate item features
    item_features = []
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Toys']
    for item_id in range(1, num_items + 1):
        features = {
            'item_id': item_id,
            'category_id': np.random.randint(1, len(categories) + 1),
            'price': np.random.uniform(10, 500),
            'popularity': np.random.randint(1, 1000),
            'rating_avg': np.random.uniform(1, 5)
        }
        item_features.append(features)
    
    item_features_df = pd.DataFrame(item_features)
    
    return interactions_df, user_features_df, item_features_df
