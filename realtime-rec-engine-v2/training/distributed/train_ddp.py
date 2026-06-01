"""
Distributed training script using PyTorch DDP for recommendation model.
Supports multi-GPU training, mixed precision, checkpointing, and monitoring.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import Config, TrainingConfig
from training.distributed.model import TwoTowerRecommendationModel, DistributedModelWrapper
from training.distributed.dataset import (
    InteractionDataset, 
    create_distributed_dataloader,
    load_sample_data,
    DataPreprocessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed trainer for recommendation model."""
    
    def __init__(self, 
                 config: Config,
                 local_rank: int,
                 world_size: int):
        
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.global_rank = config.training.rank
        
        # Setup distributed training
        self._setup_distributed()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize model
        self._initialize_model()
        
        # Initialize optimizer and scheduler
        self._initialize_optimizer()
        
        # Initialize data loaders
        self._initialize_data_loaders()
        
        # Initialize logging
        self._initialize_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Mixed precision scaler
        self.scaler = GradScaler()
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = self.config.training.master_addr
        os.environ['MASTER_PORT'] = self.config.training.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.training.backend,
            world_size=self.world_size,
            rank=self.global_rank
        )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        logger.info(f"Initialized process group: rank {self.global_rank}, world_size {self.world_size}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.checkpoint_dir = Path(self.config.model.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_model(self):
        """Initialize the recommendation model."""
        self.model = TwoTowerRecommendationModel(
            user_vocab_size=100000,  # Will be updated based on data
            item_vocab_size=500000,  # Will be updated based on data
            category_vocab_size=100,
            embedding_dim=self.config.model.embedding_dim,
            hidden_dims=self.config.model.hidden_layers,
            dropout_rate=self.config.model.dropout_rate,
            temperature=0.1
        )
        
        # Wrap with distributed wrapper
        self.model_wrapper = DistributedModelWrapper(
            model=self.model,
            local_rank=self.local_rank
        )
        
        logger.info(f"Model initialized on device {self.device}")
    
    def _initialize_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Separate parameters for different learning rates
        embedding_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'embedding' in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': embedding_params, 'lr': self.config.model.learning_rate * 0.1},
            {'params': other_params, 'lr': self.config.model.learning_rate}
        ], weight_decay=self.config.model.l2_regularization)
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.model.num_epochs,
            eta_min=self.config.model.learning_rate * 0.01
        )
        
        # Gradient scheduler for plateau detection
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def _initialize_data_loaders(self):
        """Initialize training and validation data loaders."""
        # For now, use sample data. In production, load from actual data sources
        interactions_df, user_features_df, item_features_df = load_sample_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        interactions_df = preprocessor.preprocess_interactions(interactions_df)
        
        # Update model vocab sizes based on data
        self.model.user_tower.user_embedding = torch.nn.Embedding(
            len(user_features_df) + 1, 
            self.config.model.embedding_dim
        )
        self.model.item_tower.item_embedding = torch.nn.Embedding(
            len(item_features_df) + 1, 
            self.config.model.embedding_dim
        )
        self.model.item_tower.category_embedding = torch.nn.Embedding(
            item_features_df['category_id'].nunique() + 1,
            self.config.model.embedding_dim // 4
        )
        
        # Split data
        train_size = int(0.8 * len(interactions_df))
        val_size = int(0.1 * len(interactions_df))
        
        train_interactions = interactions_df[:train_size]
        val_interactions = interactions_df[train_size:train_size + val_size]
        
        # Create datasets
        train_dataset = InteractionDataset(
            interactions_df=train_interactions,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            num_negatives=self.config.model.num_negatives,
            negative_sampling_strategy=self.config.model.negative_sampling_strategy
        )
        
        val_dataset = InteractionDataset(
            interactions_df=val_interactions,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            num_negatives=self.config.model.num_negatives,
            negative_sampling_strategy=self.config.model.negative_sampling_strategy
        )
        
        # Create distributed data loaders
        self.train_loader = create_distributed_dataloader(
            dataset=train_dataset,
            batch_size=self.config.model.batch_size,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.persistent_workers,
            shuffle=True
        )
        
        self.val_loader = create_distributed_dataloader(
            dataset=val_dataset,
            batch_size=self.config.model.batch_size,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.persistent_workers,
            shuffle=False
        )
        
        logger.info(f"Data loaders initialized: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    def _initialize_logging(self):
        """Initialize logging and monitoring."""
        if self.global_rank == 0:
            # TensorBoard logging
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
            # MLflow tracking
            mlflow.set_experiment("recommendation-engine-ddp")
            mlflow.start_run(run_name=f"ddp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Log hyperparameters
            mlflow.log_params({
                'embedding_dim': self.config.model.embedding_dim,
                'batch_size': self.config.model.batch_size,
                'learning_rate': self.config.model.learning_rate,
                'num_negatives': self.config.model.num_negatives,
                'world_size': self.world_size
            })
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            metrics = self.model_wrapper.train_step(
                batch=batch,
                optimizer=self.optimizer,
                scaler=self.scaler
            )
            
            total_loss += metrics['loss']
            self.global_step += 1
            
            # Log metrics
            if self.global_rank == 0 and batch_idx % 100 == 0:
                step_metrics = {f'train/{k}': v for k, v in metrics.items()}
                step_metrics['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                # TensorBoard logging
                for key, value in step_metrics.items():
                    self.writer.add_scalar(key, value, self.global_step)
                
                # MLflow logging
                mlflow.log_metrics(step_metrics, step=self.global_step)
                
                logger.info(f"Epoch {self.current_epoch}, Step {self.global_step}, Loss: {metrics['loss']:.4f}")
        
        # Average metrics across epoch
        epoch_metrics['train/loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {}
        total_loss = 0.0
        total_accuracy = 0.0
        total_recall_1 = 0.0
        total_recall_5 = 0.0
        total_recall_10 = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.model_wrapper.validate_step(batch=batch)
                
                total_loss += metrics['loss']
                total_accuracy += metrics.get('accuracy', 0.0)
                total_recall_1 += metrics.get('recall@1', 0.0)
                total_recall_5 += metrics.get('recall@5', 0.0)
                total_recall_10 += metrics.get('recall@10', 0.0)
        
        # Average metrics
        val_metrics = {
            'val/loss': total_loss / num_batches,
            'val/accuracy': total_accuracy / num_batches,
            'val/recall@1': total_recall_1 / num_batches,
            'val/recall@5': total_recall_5 / num_batches,
            'val/recall@10': total_recall_10 / num_batches
        }
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if self.global_rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_path)
        
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['metrics'].get('val/recall@10', 0.0)
        
        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting distributed training")
        
        try:
            for epoch in range(self.current_epoch, self.config.model.num_epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler
                self.train_loader.sampler.set_epoch(epoch)
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = self.validate()
                
                # Learning rate scheduling
                self.scheduler.step()
                self.plateau_scheduler.step(val_metrics['val/recall@10'])
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log epoch metrics
                if self.global_rank == 0:
                    for key, value in all_metrics.items():
                        self.writer.add_scalar(key, value, epoch)
                    mlflow.log_metrics(all_metrics, step=epoch)
                
                # Check for improvement
                current_metric = val_metrics['val/recall@10']
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1
                    is_best = False
                
                # Save checkpoint
                if epoch % self.config.training.save_every_n_steps == 0 or is_best:
                    self.save_checkpoint(epoch, all_metrics, is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.model.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                logger.info(f"Epoch {epoch} completed - Val Loss: {val_metrics['val/loss']:.4f}, "
                          f"Val Recall@10: {val_metrics['val/recall@10']:.4f}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Cleanup
            if self.global_rank == 0:
                self.writer.close()
                mlflow.end_run()
            
            # Clean up distributed training
            dist.destroy_process_group()
            
            logger.info("Training completed")


def setup_environment():
    """Setup environment variables and configurations."""
    # Set CUDA settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)


def train_worker(local_rank: int, world_size: int, config: Config):
    """Worker function for distributed training."""
    try:
        trainer = DistributedTrainer(
            config=config,
            local_rank=local_rank,
            world_size=world_size
        )
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed on rank {local_rank}: {str(e)}")
        raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Distributed Recommendation Model Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(),
                       help='Number of GPUs to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    if args.debug:
        config.debug = True
        config.model.num_epochs = 2
        config.model.batch_size = 32
    
    # Update training config
    config.training.world_size = args.world_size
    
    # Setup environment
    setup_environment()
    
    if args.world_size > 1:
        # Multi-GPU training
        logger.info(f"Starting distributed training on {args.world_size} GPUs")
        mp.spawn(
            train_worker,
            args=(args.world_size, config),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        logger.info("Starting single GPU training")
        train_worker(0, 1, config)


if __name__ == "__main__":
    main()
