"""
Benchmark ANN index performance including latency, recall, and throughput.
Supports A/B testing between different index configurations and algorithms.
"""

import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

from index.build_index import BaseANNIndex, IndexManager, IndexConfig, ScaNNIndex, FAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    dataset_sizes: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    embedding_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    num_queries: int = 1000
    num_threads: int = mp.cpu_count()
    warmup_queries: int = 100
    output_dir: str = "benchmark_results"
    
    # Index configurations to test
    index_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'index_type': 'faiss', 'faiss_index_type': 'IVF_PQ'},
        {'index_type': 'faiss', 'faiss_index_type': 'HNSW'},
        {'index_type': 'scann', 'scann_num_leaves': 50},
        {'index_type': 'scann', 'scann_num_leaves': 100},
    ])


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    dataset_size: int
    embedding_dim: int
    index_type: str
    index_config: Dict[str, Any]
    build_time_seconds: float
    index_size_mb: float
    
    # Search metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    queries_per_second: float
    
    # Recall metrics
    recall_at_k: Dict[int, float]
    
    # Additional metrics
    memory_usage_mb: float
    cpu_usage_percent: float


class GroundTruthCalculator:
    """Calculate ground truth results using exact search."""
    
    @staticmethod
    def exact_search(query: np.ndarray, dataset: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """Perform exact nearest neighbor search."""
        # Compute cosine similarity
        query_norm = query / np.linalg.norm(query)
        dataset_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
        similarities = np.dot(dataset_norm, query_norm)
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()


class ANNBenchmark:
    """Comprehensive ANN benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark across all configurations."""
        logger.info("Starting comprehensive ANN benchmark")
        
        for dataset_size in self.config.dataset_sizes:
            for embedding_dim in self.config.embedding_dims:
                logger.info(f"Benchmarking dataset: {dataset_size} items, {embedding_dim} dimensions")
                
                # Generate dataset
                dataset, queries = self._generate_dataset(dataset_size, embedding_dim)
                
                # Calculate ground truth
                ground_truth = self._calculate_ground_truth(dataset, queries)
                
                for index_config in self.config.index_configs:
                    logger.info(f"Testing index config: {index_config}")
                    
                    try:
                        result = self._benchmark_index_config(
                            dataset, queries, ground_truth, 
                            dataset_size, embedding_dim, index_config
                        )
                        self.results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Failed to benchmark {index_config}: {e}")
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _generate_dataset(self, dataset_size: int, embedding_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic dataset for benchmarking."""
        np.random.seed(42)  # For reproducibility
        
        # Generate dataset with some structure (clusters)
        dataset = []
        num_clusters = min(10, dataset_size // 100)
        
        for i in range(dataset_size):
            cluster_id = i % num_clusters
            cluster_center = np.random.randn(embedding_dim) * 2
            point = cluster_center + np.random.randn(embedding_dim) * 0.5
            dataset.append(point)
        
        dataset = np.array(dataset, dtype=np.float32)
        
        # Generate queries
        queries = np.random.randn(self.config.num_queries, embedding_dim).astype(np.float32)
        
        return dataset, queries
    
    def _calculate_ground_truth(self, dataset: np.ndarray, queries: np.ndarray) -> List[Tuple[List[int], List[float]]]:
        """Calculate ground truth results for all queries."""
        logger.info("Calculating ground truth...")
        
        ground_truth = []
        max_k = max(self.config.k_values)
        
        for i, query in enumerate(tqdm(queries, desc="Ground truth")):
            indices, scores = GroundTruthCalculator.exact_search(query, dataset, max_k)
            ground_truth.append((indices, scores))
        
        return ground_truth
    
    def _benchmark_index_config(self, dataset: np.ndarray, queries: np.ndarray,
                               ground_truth: List[Tuple[List[int], List[float]]],
                               dataset_size: int, embedding_dim: int,
                               index_config: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark a specific index configuration."""
        
        # Create index config
        config = IndexConfig(
            index_type=index_config['index_type'],
            embedding_dim=embedding_dim,
            **{k: v for k, v in index_config.items() if k != 'index_type'}
        )
        
        # Create index
        if config.index_type == 'scann':
            index = ScaNNIndex(config)
        else:
            index = FAISSIndex(config)
        
        # Build index
        start_time = time.time()
        item_ids = [f'item_{i}' for i in range(dataset_size)]
        
        success = index.build(dataset, item_ids)
        if not success:
            raise RuntimeError("Failed to build index")
        
        build_time = time.time() - start_time
        
        # Get index size
        index_size_mb = index.get_metrics().index_size_mb
        
        # Warmup
        self._warmup_index(index, queries[:self.config.warmup_queries])
        
        # Benchmark search performance
        latency_results, recall_results = self._benchmark_search_performance(
            index, queries, ground_truth
        )
        
        # Benchmark throughput
        qps = self._benchmark_throughput(index, queries)
        
        # Memory usage
        memory_usage_mb = self._get_memory_usage()
        
        return BenchmarkResult(
            dataset_size=dataset_size,
            embedding_dim=embedding_dim,
            index_type=config.index_type,
            index_config=index_config,
            build_time_seconds=build_time,
            index_size_mb=index_size_mb,
            latency_p50_ms=np.percentile(latency_results, 50),
            latency_p95_ms=np.percentile(latency_results, 95),
            latency_p99_ms=np.percentile(latency_results, 99),
            queries_per_second=qps,
            recall_at_k=recall_results,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=0.0  # Could be measured with psutil
        )
    
    def _warmup_index(self, index: BaseANNIndex, queries: np.ndarray):
        """Warm up the index with some queries."""
        for query in queries:
            try:
                index.search(query, k=10)
            except Exception:
                pass
    
    def _benchmark_search_performance(self, index: BaseANNIndex, queries: np.ndarray,
                                     ground_truth: List[Tuple[List[int], List[float]]]) -> Tuple[List[float], Dict[int, float]]:
        """Benchmark search latency and recall."""
        latencies = []
        recall_at_k = {k: [] for k in self.config.k_values}
        
        for i, query in enumerate(tqdm(queries, desc="Search benchmark")):
            # Measure latency
            start_time = time.time()
            try:
                result_ids, result_scores = index.search(query, max(self.config.k_values))
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Calculate recall
                gt_indices, _ = ground_truth[i]
                
                for k in self.config.k_values:
                    # Convert result IDs to indices (assuming sequential IDs)
                    result_indices = [int(id_.split('_')[1]) for id_ in result_ids[:k]]
                    gt_k = set(gt_indices[:k])
                    result_k = set(result_indices[:k])
                    
                    if len(gt_k) > 0:
                        recall = len(gt_k.intersection(result_k)) / len(gt_k)
                        recall_at_k[k].append(recall)
                    
            except Exception as e:
                logger.warning(f"Search failed for query {i}: {e}")
                latencies.append(float('inf'))
        
        # Calculate average recall
        avg_recall = {}
        for k in self.config.k_values:
            if recall_at_k[k]:
                avg_recall[k] = np.mean(recall_at_k[k])
            else:
                avg_recall[k] = 0.0
        
        return latencies, avg_recall
    
    def _benchmark_throughput(self, index: BaseANNIndex, queries: np.ndarray) -> float:
        """Benchmark queries per second."""
        num_queries = min(1000, len(queries))
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = []
            
            for query in queries[:num_queries]:
                future = executor.submit(index.search, query, k=10)
                futures.append(future)
            
            # Wait for all queries to complete
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass
        
        elapsed_time = time.time() - start_time
        qps = num_queries / elapsed_time
        
        return qps
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                'dataset_size': result.dataset_size,
                'embedding_dim': result.embedding_dim,
                'index_type': result.index_type,
                'index_config': result.index_config,
                'build_time_seconds': result.build_time_seconds,
                'index_size_mb': result.index_size_mb,
                'latency_p50_ms': result.latency_p50_ms,
                'latency_p95_ms': result.latency_p95_ms,
                'latency_p99_ms': result.latency_p99_ms,
                'queries_per_second': result.queries_per_second,
                'recall_at_k': result.recall_at_k,
                'memory_usage_mb': result.memory_usage_mb,
                'cpu_usage_percent': result.cpu_usage_percent
            })
        
        with open(os.path.join(self.config.output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_data)
        df.to_csv(os.path.join(self.config.output_dir, 'benchmark_results.csv'), index=False)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        # Load results
        df = pd.read_csv(os.path.join(self.config.output_dir, 'benchmark_results.csv'))
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Latency vs Dataset Size
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency P95 vs Dataset Size
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            axes[0, 0].plot(subset['dataset_size'], subset['latency_p95_ms'], 
                           marker='o', label=index_type)
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('P95 Latency (ms)')
        axes[0, 0].set_title('P95 Latency vs Dataset Size')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # QPS vs Dataset Size
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            axes[0, 1].plot(subset['dataset_size'], subset['queries_per_second'], 
                           marker='o', label=index_type)
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Queries per Second')
        axes[0, 1].set_title('Throughput vs Dataset Size')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Recall@10 vs Latency P95
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            # Extract recall@10 from recall_at_k column
            recall_10 = []
            for recall_str in subset['recall_at_k']:
                recall_dict = eval(recall_str)  # Parse string representation
                recall_10.append(recall_dict.get(10, 0))
            
            axes[1, 0].scatter(recall_10, subset['latency_p95_ms'], 
                              label=index_type, s=50, alpha=0.7)
        
        axes[1, 0].set_xlabel('Recall@10')
        axes[1, 0].set_ylabel('P95 Latency (ms)')
        axes[1, 0].set_title('Recall vs Latency Trade-off')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Index Size vs Dataset Size
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            axes[1, 1].plot(subset['dataset_size'], subset['index_size_mb'], 
                           marker='o', label=index_type)
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Index Size (MB)')
        axes[1, 1].set_title('Index Size vs Dataset Size')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'benchmark_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed recall analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Recall@K for different K values
        k_values = [1, 5, 10, 50, 100]
        for k in k_values:
            recalls = []
            for _, row in df.iterrows():
                recall_dict = eval(row['recall_at_k'])
                recalls.append(recall_dict.get(k, 0))
            
            axes[0].plot(df['dataset_size'], recalls, marker='o', label=f'Recall@{k}')
        
        axes[0].set_xlabel('Dataset Size')
        axes[0].set_ylabel('Recall')
        axes[0].set_title('Recall@K vs Dataset Size')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True)
        
        # Build time vs Dataset Size
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            axes[1].plot(subset['dataset_size'], subset['build_time_seconds'], 
                       marker='o', label=index_type)
        
        axes[1].set_xlabel('Dataset Size')
        axes[1].set_ylabel('Build Time (seconds)')
        axes[1].set_title('Index Build Time vs Dataset Size')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)
        
        # Memory usage vs Dataset Size
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            axes[2].plot(subset['dataset_size'], subset['memory_usage_mb'], 
                       marker='o', label=index_type)
        
        axes[2].set_xlabel('Dataset Size')
        axes[2].set_ylabel('Memory Usage (MB)')
        axes[2].set_title('Memory Usage vs Dataset Size')
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'detailed_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.config.output_dir}")
    
    def generate_report(self) -> str:
        """Generate a text report with key findings."""
        if not self.results:
            return "No benchmark results available"
        
        df = pd.DataFrame([{
            'dataset_size': r.dataset_size,
            'embedding_dim': r.embedding_dim,
            'index_type': r.index_type,
            'latency_p95_ms': r.latency_p95_ms,
            'qps': r.queries_per_second,
            'recall_at_10': r.recall_at_k.get(10, 0),
            'index_size_mb': r.index_size_mb
        } for r in self.results])
        
        report = []
        report.append("# ANN Index Benchmark Report\n")
        
        # Overall best performers
        report.append("## Best Performers by Metric\n")
        
        # Best latency
        best_latency = df.loc[df['latency_p95_ms'].idxmin()]
        report.append(f"**Best P95 Latency**: {best_latency['index_type']} "
                    f"({best_latency['latency_p95_ms']:.2f}ms) "
                    f"at {best_latency['dataset_size']} items\n")
        
        # Best throughput
        best_qps = df.loc[df['qps'].idxmax()]
        report.append(f"**Best Throughput**: {best_qps['index_type']} "
                    f"({best_qps['qps']:.0f} QPS) "
                    f"at {best_qps['dataset_size']} items\n")
        
        # Best recall
        best_recall = df.loc[df['recall_at_10'].idxmax()]
        report.append(f"**Best Recall@10**: {best_recall['index_type']} "
                    f"({best_recall['recall_at_10']:.3f}) "
                    f"at {best_recall['dataset_size']} items\n")
        
        # Smallest index
        smallest_index = df.loc[df['index_size_mb'].idxmin()]
        report.append(f"**Smallest Index**: {smallest_index['index_type']} "
                    f"({smallest_index['index_size_mb']:.1f}MB) "
                    f"at {smallest_index['dataset_size']} items\n")
        
        # Trade-off analysis
        report.append("\n## Trade-off Analysis\n")
        
        # Calculate efficiency scores
        df['efficiency_score'] = (df['recall_at_10'] * df['qps']) / (df['latency_p95_ms'] * df['index_size_mb'])
        best_efficiency = df.loc[df['efficiency_score'].idxmax()]
        report.append(f"**Best Overall Efficiency**: {best_efficiency['index_type']} "
                    f"(score: {best_efficiency['efficiency_score']:.2e})\n")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        # For low latency
        low_latency = df[df['dataset_size'] == df['dataset_size'].max()].loc[df['latency_p95_ms'].idxmin()]
        report.append(f"- **For low latency requirements**: Use {low_latency['index_type']} "
                    f"({low_latency['latency_p95_ms']:.2f}ms P95)\n")
        
        # For high throughput
        high_throughput = df[df['dataset_size'] == df['dataset_size'].max()].loc[df['qps'].idxmax()]
        report.append(f"- **For high throughput**: Use {high_throughput['index_type']} "
                    f"({high_throughput['qps']:.0f} QPS)\n")
        
        # For high recall
        high_recall = df[df['dataset_size'] == df['dataset_size'].max()].loc[df['recall_at_10'].idxmax()]
        report.append(f"- **For high recall**: Use {high_recall['index_type']} "
                    f"({high_recall['recall_at_10']:.3f} Recall@10)\n")
        
        # For memory efficiency
        memory_efficient = df[df['dataset_size'] == df['dataset_size'].max()].loc[df['index_size_mb'].idxmin()]
        report.append(f"- **For memory efficiency**: Use {memory_efficient['index_type']} "
                    f"({memory_efficient['index_size_mb']:.1f}MB)\n")
        
        report_text = ''.join(report)
        
        # Save report
        with open(os.path.join(self.config.output_dir, 'benchmark_report.md'), 'w') as f:
            f.write(report_text)
        
        return report_text


# Example usage
def main():
    """Example usage of ANN benchmarking."""
    # Create benchmark config
    config = BenchmarkConfig(
        dataset_sizes=[1000, 10000],  # Smaller for quick demo
        embedding_dims=[64, 128],
        num_queries=100,
        output_dir="benchmark_results"
    )
    
    # Run benchmark
    benchmark = ANNBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # Generate report
    report = benchmark.generate_report()
    print(report)
    
    print(f"Benchmark completed with {len(results)} results")
    print(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
