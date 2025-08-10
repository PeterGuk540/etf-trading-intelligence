"""Training orchestration for neural network models"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import wandb
import mlflow
from datetime import datetime


logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 60
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length]
        )


class CombinedLoss(nn.Module):
    """Combined loss function with multiple objectives"""
    
    def __init__(
        self,
        mse_weight: float = 0.4,
        directional_weight: float = 0.3,
        sharpe_weight: float = 0.3
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        self.sharpe_weight = sharpe_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # MSE Loss
        mse = self.mse_loss(predictions, targets)
        
        # Directional Accuracy Loss
        pred_direction = torch.sign(predictions)
        true_direction = torch.sign(targets)
        directional_loss = 1.0 - (pred_direction == true_direction).float().mean()
        
        # Sharpe Ratio Loss (negative for maximization)
        returns = predictions.squeeze()
        sharpe = returns.mean() / (returns.std() + 1e-8)
        sharpe_loss = -sharpe
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse +
            self.directional_weight * directional_loss +
            self.sharpe_weight * sharpe_loss
        )
        
        return total_loss


class TradingModel(pl.LightningModule):
    """PyTorch Lightning wrapper for trading models"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = CombinedLoss(
            mse_weight=config['loss']['mse_weight'],
            directional_weight=config['loss']['directional_weight'],
            sharpe_weight=config['loss']['sharpe_weight']
        )
        self.save_hyperparameters(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_mse', nn.MSELoss()(y_hat, y), on_step=False, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calculate additional metrics
        mse = nn.MSELoss()(y_hat, y)
        mae = nn.L1Loss()(y_hat, y)
        
        # Directional accuracy
        pred_direction = torch.sign(y_hat)
        true_direction = torch.sign(y)
        direction_acc = (pred_direction == true_direction).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_direction_acc', direction_acc, on_step=False, on_epoch=True)
        
        return loss
        
    def configure_optimizers(self):
        # Optimizer
        optimizer_config = self.config['optimizer']
        
        if optimizer_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay'],
                betas=optimizer_config['betas']
            )
        elif optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=optimizer_config['learning_rate']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
            
        # Scheduler
        scheduler_config = self.config['scheduler']
        
        if scheduler_config['type'] == 'cosine_annealing_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config['T_mult'],
                eta_min=scheduler_config['eta_min']
            )
        elif scheduler_config['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            scheduler = None
            
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        else:
            return optimizer


class TrainingOrchestrator:
    """Orchestrate the training process"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_experiment_tracking()
        
    def setup_experiment_tracking(self):
        """Setup experiment tracking (MLflow/W&B)"""
        tracking = self.config['logging']['experiment_tracking']
        
        if tracking == 'mlflow':
            mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'file:./mlruns'))
            mlflow.set_experiment(self.config.get('experiment_name', 'etf_trading'))
        elif tracking == 'wandb':
            wandb.init(
                project=self.config.get('project_name', 'etf-trading'),
                config=self.config
            )
            
    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        sequence_length: int = 60
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders"""
        
        # Convert to numpy
        X = features.values
        y = targets.values
        
        # Split data
        train_size = int(len(X) * 0.6)
        val_size = int(len(X) * 0.2)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
        
        # Create loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Train the model"""
        
        # Wrap model
        lightning_model = TradingModel(model, self.config['training'])
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config['training']['epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=True,
            gradient_clip_val=self.config['training']['regularization']['gradient_clipping'],
            precision=16 if self.config['hardware']['mixed_precision'] else 32,
            log_every_n_steps=self.config['logging']['log_interval']
        )
        
        # Train
        trainer.fit(lightning_model, train_loader, val_loader)
        
        # Test if test loader provided
        results = {}
        if test_loader:
            test_results = trainer.test(lightning_model, test_loader)
            results['test'] = test_results
            
        # Save model
        model_path = Path(self.config['logging']['checkpoint_dir']) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), model_path)
        results['model_path'] = str(model_path)
        
        return results
        
    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['regularization']['early_stopping_patience'],
                mode='min'
            )
        )
        
        # Model checkpoint
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=self.config['logging']['checkpoint_dir'],
                filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3
            )
        )
        
        # Learning rate monitor
        callbacks.append(
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        )
        
        return callbacks
        
    def adversarial_training_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """Adversarial training for robustness"""
        
        # Generate adversarial examples
        x.requires_grad = True
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Create adversarial perturbation
        data_grad = x.grad.data
        perturbation = epsilon * data_grad.sign()
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, -3, 3)  # Clamp to reasonable range
        
        # Forward pass on adversarial examples
        output_adv = model(x_adv)
        loss_adv = nn.MSELoss()(output_adv, y)
        
        return loss_adv