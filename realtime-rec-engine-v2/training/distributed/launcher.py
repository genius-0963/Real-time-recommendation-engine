"""
Launcher script for distributed training on various platforms.
Supports local multi-GPU, SLURM, and cloud-based training.
"""

import os
import sys
import subprocess
import argparse
import logging
from typing import List, Optional
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingLauncher:
    """Launcher for distributed training across different platforms."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load training configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def launch_local(self, num_gpus: int = None):
        """Launch training on local multi-GPU machine."""
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        logger.info(f"Launching local training on {num_gpus} GPUs")
        
        # Set environment variables
        env = os.environ.copy()
        env['WORLD_SIZE'] = str(num_gpus)
        
        # Build command
        cmd = [
            sys.executable,
            '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(num_gpus),
            '--master_port', str(self.config.get('master_port', '12355')),
            'train_ddp.py',
            '--config', self.config_path
        ]
        
        # Run training
        subprocess.run(cmd, env=env)
    
    def launch_slurm(self, partition: str = 'gpu', nodes: int = 1, gpus_per_node: int = 8):
        """Launch training on SLURM cluster."""
        logger.info(f"Launching SLURM training: {nodes} nodes, {gpus_per_node} GPUs per node")
        
        # Create SLURM script
        slurm_script = self._create_slurm_script(partition, nodes, gpus_per_node)
        
        # Submit job
        subprocess.run(['sbatch', slurm_script])
    
    def _create_slurm_script(self, partition: str, nodes: int, gpus_per_node: int) -> str:
        """Create SLURM batch script."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name=rec-engine-training
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source activate rec-engine

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT={self.config.get('master_port', '12355')}
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Launch training
srun python train_ddp.py --config {self.config_path}
"""
        
        script_path = 'slurm_train.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def launch_kubernetes(self, namespace: str = 'default', job_name: str = 'rec-engine-training'):
        """Launch training on Kubernetes cluster."""
        logger.info(f"Launching Kubernetes training job: {job_name}")
        
        # Create Kubernetes job manifest
        job_manifest = self._create_k8s_job_manifest(namespace, job_name)
        
        # Apply manifest
        subprocess.run(['kubectl', 'apply', '-f', '-'], input=job_manifest.encode())
    
    def _create_k8s_job_manifest(self, namespace: str, job_name: str) -> str:
        """Create Kubernetes job manifest."""
        num_workers = self.config.get('num_workers', 4)
        
        manifest = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
spec:
  parallelism: {num_workers}
  completions: {num_workers}
  template:
    metadata:
      name: {job_name}-worker
    spec:
      containers:
      - name: training-container
        image: rec-engine/training:latest
        command: ["python", "train_ddp.py", "--config", "/config/config.yaml"]
        env:
        - name: MASTER_ADDR
          value: "{job_name}-master-0.{job_name}-master.{namespace}.svc.cluster.local"
        - name: MASTER_PORT
          value: "{self.config.get('master_port', '12355')}"
        - name: WORLD_SIZE
          value: "{num_workers}"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: LOCAL_RANK
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: config-volume
          mountPath: /config
        - name: data-volume
          mountPath: /data
        - name: checkpoint-volume
          mountPath: /checkpoints
      volumes:
      - name: config-volume
        configMap:
          name: training-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: checkpoint-volume
        persistentVolumeClaim:
          claimName: checkpoint-pvc
      restartPolicy: OnFailure
"""
        return manifest
    
    def launch_aws_sagemaker(self, role_arn: str, instance_type: str = 'ml.p3.8xlarge'):
        """Launch training on AWS SageMaker."""
        logger.info(f"Launching SageMaker training on {instance_type}")
        
        import sagemaker
        from sagemaker.pytorch import PyTorch
        
        # Create SageMaker estimator
        estimator = PyTorch(
            entry_point='train_ddp.py',
            source_dir='.',
            role=role_arn,
            instance_count=1,
            instance_type=instance_type,
            framework_version='2.1.0',
            py_version='py310',
            distribution={'mpi': {'enabled': True}},
            hyperparameters={
                'config': '/opt/ml/input/data/config/config.yaml'
            }
        )
        
        # Start training
        estimator.fit({'config': self.config_path})
    
    def launch_gcp_ai_platform(self, region: str = 'us-central1', 
                              accelerator_type: str = 'NVIDIA_TESLA_V100'):
        """Launch training on Google Cloud AI Platform."""
        logger.info(f"Launching GCP AI Platform training in {region}")
        
        # Build gcloud command
        cmd = [
            'gcloud', 'ai-platform', 'jobs', 'submit', 'training',
            '--region', region,
            '--scale-tier', 'custom',
            '--master-machine-type', 'n1-standard-8',
            '--master-accelerator', f'type={accelerator_type},count=1',
            '--worker-machine-type', 'n1-standard-8',
            '--worker-accelerator', f'type={accelerator_type},count=1',
            '--worker-count', '3',
            '--runtime-version', '2.9',
            '--python-version', '3.10',
            '--distribution-strategy', 'MirroredStrategy',
            '--module-name', 'training.distributed.train_ddp',
            '--', '--config', self.config_path
        ]
        
        subprocess.run(cmd)


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Distributed Training Launcher')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--platform', type=str, choices=['local', 'slurm', 'k8s', 'sagemaker', 'gcp'],
                       default='local', help='Training platform')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs (local only)')
    parser.add_argument('--partition', type=str, default='gpu', help='SLURM partition')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes (SLURM)')
    parser.add_argument('--gpus-per-node', type=int, default=8, help='GPUs per node (SLURM)')
    parser.add_argument('--namespace', type=str, default='default', help='Kubernetes namespace')
    parser.add_argument('--job-name', type=str, default='rec-engine-training', help='Kubernetes job name')
    parser.add_argument('--role-arn', type=str, help='AWS SageMaker role ARN')
    parser.add_argument('--instance-type', type=str, default='ml.p3.8xlarge', help='SageMaker instance type')
    parser.add_argument('--region', type=str, default='us-central1', help='GCP region')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = TrainingLauncher(args.config)
    
    # Launch based on platform
    if args.platform == 'local':
        launcher.launch_local(args.num_gpus)
    elif args.platform == 'slurm':
        launcher.launch_slurm(args.partition, args.nodes, args.gpus_per_node)
    elif args.platform == 'k8s':
        launcher.launch_kubernetes(args.namespace, args.job_name)
    elif args.platform == 'sagemaker':
        if not args.role_arn:
            raise ValueError("--role-arn is required for SageMaker")
        launcher.launch_aws_sagemaker(args.role_arn, args.instance_type)
    elif args.platform == 'gcp':
        launcher.launch_gcp_ai_platform(args.region)
    else:
        raise ValueError(f"Unknown platform: {args.platform}")


if __name__ == "__main__":
    main()
