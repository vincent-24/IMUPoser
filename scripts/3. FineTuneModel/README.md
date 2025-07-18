# IMUPoser Model Fine-tuning

This directory contains scripts and tools for fine-tuning pretrained IMUPoser models on specific datasets.

## Overview

Fine-tuning allows you to adapt a pretrained IMUPoser model to perform better on specific datasets or use cases. The fine-tuning process uses a lower learning rate and continues training from a pretrained checkpoint.

## Files

- **`1. FineTune Model.py`** - Main fine-tuning script
- **`run_finetune.sh`** - Shell script wrapper for easy execution
- **`config_examples.txt`** - Example configurations for different scenarios
- **`README.md`** - This documentation

## Quick Start

### 1. List Available Checkpoints

```bash
./run_finetune.sh --list
```

### 2. Run Fine-tuning

```bash
./run_finetune.sh \
    --checkpoint "../../../Trained_Model_wandb/IMUPoserGlobalModel_acting-07072025/epoch=epoch=12-val_loss=validation_step_loss=0.00827.ckpt" \
    --experiment "my_finetune_experiment" \
    --combo_id "dip_imu"
```

### 3. Test Run (Fast Development)

```bash
./run_finetune.sh \
    --checkpoint "path/to/checkpoint.ckpt" \
    --experiment "test_run" \
    --combo_id "dip_imu" \
    --fast_dev_run
```

## Detailed Usage

### Python Script

```python
python "1. FineTune Model.py" \
    --combo_id dip_imu \
    --experiment finetune_experiment \
    --checkpoint_path path/to/checkpoint.ckpt \
    --max_epochs 100 \
    --lr 1e-4
```

### Shell Script Options

```bash
./run_finetune.sh [OPTIONS]

OPTIONS:
    -h, --help              Show help message
    -l, --list              List available checkpoints
    -c, --checkpoint PATH   Path to checkpoint file
    -e, --experiment NAME   Experiment name
    -j, --combo_id ID       Combo ID (joint set)
    --max_epochs N          Maximum epochs (default: 100)
    --lr RATE              Learning rate (default: 1e-4)
    --gpu_id ID            GPU device ID (default: 0)
    --fast_dev_run         Run fast development run
    --dry_run              Show command without executing
```

## Parameters

### Required Parameters

- **`checkpoint_path`** - Path to the pretrained model checkpoint file
- **`experiment`** - Name for the fine-tuning experiment (used in logging)
- **`combo_id`** - Joint set configuration ID (determines dataset and IMU setup)

### Optional Parameters

- **`max_epochs`** - Maximum training epochs (default: 100)
- **`lr`** - Learning rate (default: 1e-4, lower than base training)
- **`patience`** - Early stopping patience (default: 10)
- **`gpu_id`** - GPU device ID (default: "0")
- **`use_joint_loss`** - Use 3D joint position loss (default: True)
- **`loss_type`** - Loss function: 'mse' or 'l1' (default: 'mse')

### Combo IDs

Common combo IDs include:
- `dip_imu` - DIP dataset configuration
- `totalcapture` - TotalCapture dataset configuration

Check `imuposer/config.py` under `amass_combos` for all available options.

## Fine-tuning Process

### 1. Model Loading
The script loads a pretrained `IMUPoserModel` from the specified checkpoint.

### 2. Wrapper Creation
The pretrained model is wrapped in `IMUPoserModelFineTune` which:
- Freezes or unfreezes parameters as needed
- Uses a lower learning rate
- Maintains the same architecture

### 3. Dataset Selection
Based on the model type (`GlobalModelIMUPoserFineTuneDIP`), the system automatically selects the appropriate dataset for fine-tuning.

### 4. Training
Uses PyTorch Lightning with:
- Early stopping based on validation loss
- Model checkpointing (saves top 5 models)
- Weights & Biases logging
- Gradient clipping for stability

## Output

Fine-tuning creates:
- **Checkpoint files** - Saved in the experiment directory
- **Logs** - Weights & Biases logs for monitoring
- **Best model info** - Text file with paths to best models

### Output Structure
```
/path/to/output/checkpoints/<experiment_name>/
├── finetune-epoch=XX-val_loss=X.XXXXX.ckpt  # Fine-tuned checkpoints
├── best_finetune_model.txt                   # Best model info
└── wandb/                                    # Training logs
```

## Best Practices

### Learning Rate
- Use lower learning rates (1e-5 to 1e-4) for fine-tuning
- Start with 1e-4 and reduce if training is unstable
- Conservative: 1e-5, Standard: 1e-4, Aggressive: 5e-4

### Epochs
- Start with fewer epochs (30-50) to avoid overfitting
- Monitor validation loss carefully
- Use early stopping with appropriate patience

### Dataset Preparation
- Ensure your fine-tuning dataset is properly processed
- The system expects `dip_train.pt` and `dip_test.pt` files
- Data should be in the same format as the original training data

### Monitoring
- Watch the validation loss carefully
- Fine-tuning should improve performance gradually
- If validation loss increases, consider reducing learning rate

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   - Verify the checkpoint path exists
   - Use absolute paths to avoid confusion

2. **CUDA out of memory**
   - Reduce batch size in the config
   - Use a smaller model or gradient accumulation

3. **Dataset not found**
   - Ensure the fine-tuning dataset files exist
   - Check the data directory structure

4. **No improvement**
   - Try a lower learning rate
   - Increase training epochs
   - Verify dataset quality

### Debug Mode
Use `--fast_dev_run` for quick testing without full training.

## Examples

### Example 1: Standard Fine-tuning
```bash
./run_finetune.sh \
    --checkpoint "../../../Trained_Model_wandb/IMUPoserGlobalModel_acting-07072025/epoch=epoch=12-val_loss=validation_step_loss=0.00827.ckpt" \
    --experiment "dip_finetune_v1" \
    --combo_id "dip_imu" \
    --max_epochs 50 \
    --lr 1e-4
```

### Example 2: Conservative Fine-tuning
```bash
./run_finetune.sh \
    --checkpoint "path/to/checkpoint.ckpt" \
    --experiment "conservative_finetune" \
    --combo_id "dip_imu" \
    --max_epochs 30 \
    --lr 1e-5
```

### Example 3: Quick Test
```bash
./run_finetune.sh \
    --checkpoint "path/to/checkpoint.ckpt" \
    --experiment "test" \
    --combo_id "dip_imu" \
    --fast_dev_run
```

For more detailed examples, see `config_examples.txt`.
