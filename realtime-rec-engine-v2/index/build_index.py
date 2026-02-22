"""
Build and manage ANN (Approximate Nearest Neighbor) indexes using ScaNN and FAISS.
Supports distributed indexing, quantization, and hot-swapping for production deployment.
"""

import os
import logging
import time
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

# ScaNN imports
try:
    import scann
    SCANN_AVAILABLE = True
except ImportError:
    SCANN_AVAILABLE = False
    logging.warning("ScaNN not available, falling back to FAISS")

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available")

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for ANN index building."""
    index_type: str = 'scann'  # 'scann' or 'faiss'
    embedding_dim: int = 128
    num_items: int = 1000000
    index_path: str = '/tmp/ann_index'
    
    # ScaNN specific
    scann_num_leaves: int = 100
    scann_num_leaves_to_search: int = 10
    scann_quantization_bits: int = 8
    
    # FAISS specific
    faiss_index_type: str = 'IVF_PQ'  # 'IVF_PQ', 'HNSW', 'IVF_FLAT'
    faiss_nlist: int = 1000
    faiss_m: int = 64
    faiss_nbits: int = 8
    
    # General
    num_threads: int = mp.cpu_count()
    batch_size: int = 10000
    use_gpu: bool = False
    gpu_device_ids: List[int] = None
    
    def __post_init__(self):
        if self.gpu_device_ids is None:
            self.gpu_device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


@dataclass
class IndexMetrics:
    """Metrics for index performance."""
    build_time_seconds: float = 0.0
    index_size_mb: float = 0.0
    recall_at_k: Dict[int, float] = None
    latency_p95_ms: float = 0.0
    queries_per_second: float = 0.0
    
    def __post_init__(self):
        if self.recall_at_k is None:
            self.recall_at_k = {}


class BaseANNIndex:
    """Base class for ANN indexes."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.index = None
        self.item_ids = None
        self.is_built = False
        self.metrics = IndexMetrics()
        
    def build(self, embeddings: np.ndarray, item_ids: List[str]) -> bool:
        """Build the ANN index."""
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search for nearest neighbors."""
        raise NotImplementedError
    
    def save(self, path: str) -> bool:
        """Save index to disk."""
        raise NotImplementedError
    
    def load(self, path: str) -> bool:
        """Load index from disk."""
        raise NotImplementedError
    
    def get_metrics(self) -> IndexMetrics:
        """Get index performance metrics."""
        return self.metrics


class ScaNNIndex(BaseANNIndex):
    """ScaNN-based ANN index."""
    
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        
        if not SCANN_AVAILABLE:
            raise ImportError("ScaNN is not available")
        
        self.searcher = None
    
    def build(self, embeddings: np.ndarray, item_ids: List[str]) -> bool:
        """Build ScaNN index."""
        try:
            start_time = time.time()
            
            logger.info(f"Building ScaNN index with {len(embeddings)} embeddings")
            
            # Normalize embeddings
            embeddings = embeddings.astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create ScaNN searcher
            self.searcher = scann.scann_ops_pybind.builder(
                embeddings, 
                self.config.scann_num_leaves, 
                "dot_product"
            ).tree(
                num_leaves=self.config.scann_num_leaves,
                num_leaves_to_search=self.config.scann_num_leaves_to_search,
                training_sample_size=min(len(embeddings), 100000)
            ).score_ah(
                dim=self.config.embedding_dim,
                anisotropic_quantization_threshold=0.2
            ).reorder(
                num_reorder=100
            ).build()
            
            self.item_ids = item_ids
            self.is_built = True
            
            # Update metrics
            self.metrics.build_time_seconds = time.time() - start_time
            self.metrics.index_size_mb = self._get_index_size()
            
            logger.info(f"ScaNN index built in {self.metrics.build_time_seconds:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build ScaNN index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search using ScaNN."""
        if not self.is_built or self.searcher is None:
            raise RuntimeError("Index not built")
        
        try:
            # Normalize query
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search
            start_time = time.time()
            neighbors, scores = self.searcher.search(query_embedding, final_num_neighbors=k)
            
            # Convert to item IDs
            item_ids = [self.item_ids[idx] for idx in neighbors]
            scores = scores.tolist()
            
            # Update latency metric
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.latency_p95_ms = max(self.metrics.latency_p95_ms, search_time)
            
            return item_ids, scores
            
        except Exception as e:
            logger.error(f"ScaNN search failed: {e}")
            return [], []
    
    def save(self, path: str) -> bool:
        """Save ScaNN index."""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save searcher
            searcher_path = os.path.join(path, 'scann_searcher')
            self.searcher.serialize(searcher_path)
            
            # Save metadata
            metadata = {
                'item_ids': self.item_ids,
                'config': asdict(self.config),
                'metrics': asdict(self.metrics)
            }
            
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"ScaNN index saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ScaNN index: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load ScaNN index."""
        try:
            # Load metadata
            with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            
            self.item_ids = metadata['item_ids']
            self.metrics = IndexMetrics(**metadata['metrics'])
            
            # Load searcher
            searcher_path = os.path.join(path, 'scann_searcher')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_path)
            
            self.is_built = True
            logger.info(f"ScaNN index loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ScaNN index: {e}")
            return False
    
    def _get_index_size(self) -> float:
        """Get index size in MB."""
        if not self.searcher:
            return 0.0
        
        # Estimate size based on embeddings
        return len(self.item_ids) * self.config.embedding_dim * 4 / (1024 * 1024)  # float32


class FAISSIndex(BaseANNIndex):
    """FAISS-based ANN index."""
    
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available")
        
        self.index = None
        self.faiss_config = None
    
    def build(self, embeddings: np.ndarray, item_ids: List[str]) -> bool:
        """Build FAISS index."""
        try:
            start_time = time.time()
            
            logger.info(f"Building FAISS index with {len(embeddings)} embeddings")
            
            # Prepare embeddings
            embeddings = embeddings.astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create index based on type
            if self.config.faiss_index_type == 'IVF_PQ':
                self.index = self._build_ivf_pq_index(embeddings)
            elif self.config.faiss_index_type == 'HNSW':
                self.index = self._build_hnsw_index(embeddings)
            elif self.config.faiss_index_type == 'IVF_FLAT':
                self.index = self._build_ivf_flat_index(embeddings)
            else:
                raise ValueError(f"Unknown FAISS index type: {self.config.faiss_index_type}")
            
            # Train and add vectors
            if hasattr(self.index, 'train'):
                self.index.train(embeddings)
            self.index.add(embeddings)
            
            self.item_ids = item_ids
            self.is_built = True
            
            # Update metrics
            self.metrics.build_time_seconds = time.time() - start_time
            self.metrics.index_size_mb = self._get_index_size()
            
            logger.info(f"FAISS index built in {self.metrics.build_time_seconds:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def _build_ivf_pq_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build IVF+PQ index."""
        nlist = min(self.config.faiss_nlist, len(embeddings) // 10)
        m = self.config.faiss_m
        nbits = self.config.faiss_nbits
        
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, self.config.embedding_dim, nlist, m, nbits)
        
        return index
    
    def _build_hnsw_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build HNSW index."""
        index = faiss.IndexHNSWFlat(self.config.embedding_dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        
        return index
    
    def _build_ivf_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build IVF Flat index."""
        nlist = min(self.config.faiss_nlist, len(embeddings) // 10)
        
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dim, nlist)
        
        return index
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search using FAISS."""
        if not self.is_built or self.index is None:
            raise RuntimeError("Index not built")
        
        try:
            # Normalize query
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            start_time = time.time()
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert to item IDs
            item_ids = []
            valid_scores = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(self.item_ids):  # FAISS returns -1 for invalid results
                    item_ids.append(self.item_ids[idx])
                    valid_scores.append(float(score))
            
            # Update latency metric
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.latency_p95_ms = max(self.metrics.latency_p95_ms, search_time)
            
            return item_ids, valid_scores
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return [], []
    
    def save(self, path: str) -> bool:
        """Save FAISS index."""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save index
            index_path = os.path.join(path, 'faiss_index')
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata = {
                'item_ids': self.item_ids,
                'config': asdict(self.config),
                'metrics': asdict(self.metrics)
            }
            
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"FAISS index saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load FAISS index."""
        try:
            # Load metadata
            with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            
            self.item_ids = metadata['item_ids']
            self.metrics = IndexMetrics(**metadata['metrics'])
            
            # Load index
            index_path = os.path.join(path, 'faiss_index')
            self.index = faiss.read_index(index_path)
            
            self.is_built = True
            logger.info(f"FAISS index loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def _get_index_size(self) -> float:
        """Get index size in MB."""
        if not self.index:
            return 0.0
        
        # FAISS doesn't provide direct size info, estimate based on type
        base_size = len(self.item_ids) * self.config.embedding_dim * 4 / (1024 * 1024)  # float32
        
        if self.config.faiss_index_type == 'IVF_PQ':
            # PQ reduces size significantly
            return base_size * self.config.faiss_m * self.config.faiss_nbits / (8 * 32)
        elif self.config.faiss_index_type == 'HNSW':
            # HNSW adds overhead for graph structure
            return base_size * 1.5
        else:
            return base_size


class DistributedIndexBuilder:
    """Build indexes in a distributed manner."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.num_workers = config.num_threads
    
    def build_distributed(self, embeddings: np.ndarray, item_ids: List[str]) -> BaseANNIndex:
        """Build index using multiple processes."""
        try:
            # Split data into chunks
            chunk_size = len(embeddings) // self.num_workers
            chunks = []
            id_chunks = []
            
            for i in range(self.num_workers):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(embeddings)
                
                chunks.append(embeddings[start_idx:end_idx])
                id_chunks.append(item_ids[start_idx:end_idx])
            
            # Build partial indexes in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for i, (chunk, id_chunk) in enumerate(zip(chunks, id_chunks)):
                    future = executor.submit(self._build_partial_index, chunk, id_chunk, i)
                    futures.append(future)
                
                # Wait for completion
                partial_indexes = []
                for future in futures:
                    partial_index = future.result()
                    partial_indexes.append(partial_index)
            
            # Merge partial indexes
            merged_index = self._merge_indexes(partial_indexes, embeddings, item_ids)
            
            return merged_index
            
        except Exception as e:
            logger.error(f"Distributed index building failed: {e}")
            raise
    
    def _build_partial_index(self, embeddings: np.ndarray, item_ids: List[str], worker_id: int) -> BaseANNIndex:
        """Build partial index in worker process."""
        try:
            # Create index for this chunk
            if self.config.index_type == 'scann':
                index = ScaNNIndex(self.config)
            else:
                index = FAISSIndex(self.config)
            
            # Build index
            index.build(embeddings, item_ids)
            
            return index
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            raise
    
    def _merge_indexes(self, partial_indexes: List[BaseANNIndex], 
                       full_embeddings: np.ndarray, full_item_ids: List[str]) -> BaseANNIndex:
        """Merge partial indexes into final index."""
        try:
            # For now, rebuild with full data (could be optimized)
            if self.config.index_type == 'scann':
                final_index = ScaNNIndex(self.config)
            else:
                final_index = FAISSIndex(self.config)
            
            final_index.build(full_embeddings, full_item_ids)
            
            return final_index
            
        except Exception as e:
            logger.error(f"Failed to merge indexes: {e}")
            raise


class IndexManager:
    """Manage multiple ANN indexes with hot-swapping."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.active_index = None
        self.standby_index = None
        self.index_lock = threading.RLock()
        self.build_queue = queue.Queue()
        self.is_building = False
        
        # Create index directory
        os.makedirs(config.index_path, exist_ok=True)
        
    def build_index(self, embeddings: np.ndarray, item_ids: List[str], 
                   distributed: bool = False) -> bool:
        """Build new index."""
        try:
            with self.index_lock:
                if self.is_building:
                    logger.warning("Index build already in progress")
                    return False
                
                self.is_building = True
            
            logger.info(f"Building new {self.config.index_type} index")
            
            # Build index
            if distributed:
                builder = DistributedIndexBuilder(self.config)
                new_index = builder.build_distributed(embeddings, item_ids)
            else:
                if self.config.index_type == 'scann':
                    new_index = ScaNNIndex(self.config)
                else:
                    new_index = FAISSIndex(self.config)
                
                new_index.build(embeddings, item_ids)
            
            # Save index
            timestamp = int(time.time())
            index_dir = os.path.join(self.config.index_path, f'index_{timestamp}')
            new_index.save(index_dir)
            
            # Hot swap
            with self.index_lock:
                self.standby_index = new_index
                self.active_index, self.standby_index = self.standby_index, self.active_index
                
                if self.standby_index:
                    # Clean up old index
                    self._cleanup_old_index()
            
            self.is_building = False
            logger.info("New index built and activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            self.is_building = False
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search using active index."""
        with self.index_lock:
            if self.active_index is None:
                raise RuntimeError("No active index available")
            
            return self.active_index.search(query_embedding, k)
    
    def load_latest_index(self) -> bool:
        """Load the latest index from disk."""
        try:
            # Find latest index directory
            index_dirs = [d for d in os.listdir(self.config.index_path) 
                         if d.startswith('index_') and os.path.isdir(os.path.join(self.config.index_path, d))]
            
            if not index_dirs:
                logger.warning("No index directories found")
                return False
            
            latest_dir = max(index_dirs, key=lambda x: int(x.split('_')[1]))
            index_path = os.path.join(self.config.index_path, latest_dir)
            
            # Load index
            if self.config.index_type == 'scann':
                index = ScaNNIndex(self.config)
            else:
                index = FAISSIndex(self.config)
            
            if index.load(index_path):
                with self.index_lock:
                    self.active_index = index
                
                logger.info(f"Loaded index from {index_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to load latest index: {e}")
            return False
    
    def _cleanup_old_index(self):
        """Clean up old standby index."""
        if self.standby_index:
            # Could remove old files here if needed
            self.standby_index = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get index metrics."""
        with self.index_lock:
            if self.active_index:
                return {
                    'active_index_metrics': asdict(self.active_index.get_metrics()),
                    'is_building': self.is_building,
                    'index_type': self.config.index_type
                }
            else:
                return {
                    'active_index_metrics': None,
                    'is_building': self.is_building,
                    'index_type': self.config.index_type
                }


# Example usage
def main():
    """Example usage of ANN index building."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Create index config
    index_config = IndexConfig(
        index_type=config.model.index_type,
        embedding_dim=config.model.embedding_dim,
        scann_num_leaves=config.model.scann_num_leaves,
        scann_num_leaves_to_search=config.model.scann_num_leaves_to_search,
        scann_quantization_bits=config.model.scann_quantization_bits
    )
    
    # Generate sample embeddings
    num_items = 10000
    embeddings = np.random.random((num_items, index_config.embedding_dim)).astype(np.float32)
    item_ids = [f'item_{i}' for i in range(num_items)]
    
    # Create index manager
    manager = IndexManager(index_config)
    
    try:
        # Build index
        success = manager.build_index(embeddings, item_ids, distributed=False)
        logger.info(f"Index build success: {success}")
        
        # Test search
        query = np.random.random(index_config.embedding_dim).astype(np.float32)
        results, scores = manager.search(query, k=10)
        logger.info(f"Search results: {len(results)} items")
        
        # Get metrics
        metrics = manager.get_metrics()
        logger.info(f"Index metrics: {metrics}")
        
    finally:
        # Cleanup
        pass


if __name__ == "__main__":
    main()
