#!/usr/bin/env python3
"""
IMUPoser Model Fine-tuning Script

This script fine-tunes a pretrained IMUPoser model on a specific dataset.
The fine-tuning process uses the IMUPoserModelFineTune class which wraps
a pretrained model and allows for continued training on new data.

Usage:
    python "1. FineTune Model.py" --combo_id <combo_id> --experiment <experiment_name> --checkpoint_path <path_to_checkpoint>

Example:
    python "1. FineTune Model.py" --combo_id dip_imu --experiment finetune_dip --checkpoint_path ../../../Trained_Model_wandb/IMUPoserGlobalModel_acting-07072025/epoch=epoch=12-val_loss=validation_step_loss=0.00827.ckpt
"""

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import pathlib
import argparse
from pathlib import Path
import os

# Add the src directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

# Add project root to path to find constants.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.models.LSTMs.IMUPoser_Model_FineTune import IMUPoserModelFineTune
from imuposer.datasets.utils import get_datamodule
from constants import BASE_MODEL_FPATH

seed_everything(42, workers=True)

def get_finetune_parser():
    """Create argument parser for fine-tuning script."""
    parser = argparse.ArgumentParser(description='Fine-tune IMUPoser model')
    parser.add_argument('--combo_id', type=str, default="global", help='Combo ID for joint set (e.g., dip_imu, custom_set)')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name for logging')
    parser.add_argument('--checkpoint_path', type=str, default=BASE_MODEL_FPATH, help='Path to pretrained model checkpoint')
    parser.add_argument('--root_dir', type=str, default="../..", help='Root directory for data and outputs')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning (lower than base training)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU device ID')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a fast development run for testing')
    parser.add_argument('--use_joint_loss', action='store_true', default=True, help='Use joint position loss in addition to pose loss')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='Loss function type')
    parser.add_argument('--use_llm', action='store_true', default=False, help='Use LLM dataset for fine-tuning')
    return parser

def load_pretrained_model(checkpoint_path, config):
    """Load the pretrained model from checkpoint."""
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        pretrained_model = IMUPoserModel.load_from_checkpoint(
            checkpoint_path,
            config=config
        )
        print(f"Successfully loaded pretrained model")
        print(f"Model output size: {pretrained_model.n_pose_output}")
        return pretrained_model
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def setup_finetune_model(pretrained_model, config, lr):     # CREATES THE FINE-TUNED MODEL
    """Create the fine-tuning model wrapper."""
    print("Setting up fine-tuning model...")
    
    # Create the fine-tuning model (IMUPoser/src/imuposer/models/LSTMs/IMUPoser_Model_FineTune.py)
    finetune_model = IMUPoserModelFineTune(
        config=config,
        pretrained_model=pretrained_model
    )
    
    finetune_model.lr = lr
    
    print(f"Fine-tuning model created with learning rate: {lr}")
    return finetune_model

def setup_trainer(config, max_epochs, patience, fast_dev_run):
    """Setup PyTorch Lightning trainer with callbacks."""
    checkpoint_path = config.checkpoint_path
    
    wandb_logger = WandbLogger(
        project=config.experiment, 
        save_dir=checkpoint_path,
        name=f"finetune_{config.experiment}"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="validation_step_loss", 
        mode="min", 
        verbose=True,
        min_delta=0.00001, 
        patience=patience
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss", 
        mode="min", 
        verbose=True,
        save_top_k=5, 
        dirpath=checkpoint_path, 
        save_weights_only=True, 
        filename='finetune-epoch={epoch}-val_loss={validation_step_loss:.5f}'
    )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[int(config.device.index)],
        callbacks=[early_stopping_callback, checkpoint_callback],
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0, 
        check_val_every_n_epoch=1,
        log_every_n_steps=10
    )
    
    return trainer, checkpoint_callback

def main():
    """Main fine-tuning function."""
    parser = get_finetune_parser()
    args = parser.parse_args()
    
    print("="*60)
    print("IMUPoser Model Fine-tuning")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Combo ID: {args.combo_id}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"GPU: {args.gpu_id}")
    print(f"Using {'LLM' if args.use_llm else 'DIP'} dataset")
    print("="*60)
    
    base_config = Config(
        experiment=f"{args.experiment}_base",
        model="GlobalModelIMUPoser",
        project_root_dir=args.root_dir,
        joints_set=amass_combos[args.combo_id],
        normalize="no_translation",
        r6d=True,
        loss_type=args.loss_type,
        use_joint_loss=args.use_joint_loss,
        device=args.gpu_id
    )
    
    pretrained_model = load_pretrained_model(args.checkpoint_path, base_config)
    
    finetune_config = Config(
        experiment=f"{args.experiment}_finetune",
        model="GlobalModelIMUPoserFineTune", 
        project_root_dir=args.root_dir,
        joints_set=amass_combos[args.combo_id],
        normalize="no_translation",
        r6d=True,
        loss_type=args.loss_type,
        use_joint_loss=args.use_joint_loss,
        device=args.gpu_id,
        use_llm=args.use_llm
    )
    
    finetune_model = setup_finetune_model(pretrained_model, finetune_config, args.lr)   # TAKES finetune_config TO SET THE MODEL
    
    print("Setting up data module...")
    datamodule = get_datamodule(finetune_config, use_llm=args.use_llm)
    print(f"Data module created for model type: {finetune_config.model}")
    
    trainer, checkpoint_callback = setup_trainer(
        finetune_config, 
        args.max_epochs, 
        args.patience, 
        args.fast_dev_run
    )
    
    print("\nStarting fine-tuning...")
    print(f"Output directory: {finetune_config.checkpoint_path}")
    
    try:
        trainer.fit(finetune_model, datamodule=datamodule)
        
        best_model_path = finetune_config.checkpoint_path / "best_finetune_model.txt"
        with open(best_model_path, "w") as f:
            f.write(f"Fine-tuning completed successfully!\n")
            f.write(f"Base model: {args.checkpoint_path}\n")
            f.write(f"Best fine-tuned model: {checkpoint_callback.best_model_path}\n\n")
            f.write(f"All saved models:\n{checkpoint_callback.best_k_models}\n")
        
        print(f"\nFine-tuning completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Results saved to: {best_model_path}")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        raise
    
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user")
        return

if __name__ == "__main__":
    main()
