#!/bin/bash

# IMUPoser Fine-tuning Runner Script
# This script provides examples and easy ways to run fine-tuning

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/1. FineTune Model.py"
# Change to project root before importing constants
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_ROOT" && python -c "import sys; sys.path.append('.'); import constants; print(constants.PROJECT_ROOT_DIR)")"
TRAINED_MODELS_DIR="$SCRIPT_DIR/$(cd "$PROJECT_ROOT" && python -c 'import sys; sys.path.append("."); import constants; print(constants.BASE_MODEL_DIR)')"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to list available checkpoints
list_checkpoints() {
    print_info "Available trained models:"
    if [ -d "$TRAINED_MODELS_DIR" ]; then
        for model_dir in "$TRAINED_MODELS_DIR"/*; do
            if [ -d "$model_dir" ]; then
                model_name=$(basename "$model_dir")
                echo "  📁 $model_name"
                
                # List checkpoint files
                for ckpt in "$model_dir"/*.ckpt; do
                    if [ -f "$ckpt" ]; then
                        ckpt_name=$(basename "$ckpt")
                        echo "    📄 $ckpt_name"
                    fi
                done
                echo
            fi
        done
    else
        print_warning "Trained models directory not found: $TRAINED_MODELS_DIR"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
IMUPoser Fine-tuning Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -l, --list              List available checkpoints
    -c, --checkpoint PATH   Path to checkpoint file
    -e, --experiment NAME   Experiment name
    -j, --combo_id ID       Combo ID (joint set)
    -r, --root_dir PATH     Root directory (default: $ROOT_DIR)
    --max_epochs N          Maximum epochs (default: 100)
    --lr RATE              Learning rate (default: 1e-4)
    --gpu_id ID            GPU device ID (default: 0)
    --fast_dev_run         Run fast development run
    --dry_run              Show command without executing
    --use_llm, --use-llm    Use LLM dataset for fine-tuning (default: false)

EXAMPLES:
    # List available checkpoints
    $0 --list

    # Fine-tune on DIP dataset
    $0 --checkpoint "$TRAINED_MODELS_DIR/IMUPoserGlobalModel_acting-07072025/epoch=epoch=12-val_loss=validation_step_loss=0.00827.ckpt" \\
       --experiment "finetune_acting_on_dip" \\
       --combo_id "dip_imu"

    # Quick test run
    $0 --checkpoint "path/to/checkpoint.ckpt" \\
       --experiment "test_finetune" \\
       --combo_id "dip_imu" \\
       --fast_dev_run

COMBO IDs:
    Common combo IDs include:
    - dip_imu: DIP dataset configuration
    - totalcapture: TotalCapture dataset configuration
    - custom: Custom joint set configuration
    
    Check the amass_combos in imuposer/config.py for all available options.

EOF
}

# Parse command line arguments
CHECKPOINT="$(cd "$PROJECT_ROOT" && python -c "import sys; sys.path.append('.'); import constants; print(constants.BASE_MODEL_FPATH)")"
EXPERIMENT=""
COMBO_ID="global"
MAX_EPOCHS=100
LR="1e-4"
GPU_ID="0"
FAST_DEV_RUN=""
DRY_RUN=false
USE_LLM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            list_checkpoints
            exit 0
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -j|--combo_id)
            COMBO_ID="$2"
            shift 2
            ;;
        -r|--root_dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --fast_dev_run)
            FAST_DEV_RUN="--fast_dev_run"
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --use_llm|--use-llm)
            USE_LLM=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments - only checkpoint is required if not using defaults
if [ -z "$CHECKPOINT" ]; then
    # Try to find a default checkpoint
    DEFAULT_CHECKPOINT="$SCRIPT_DIR/$(cd "$PROJECT_ROOT" && python -c 'import sys; sys.path.append("."); import constants; print(constants.BASE_MODEL_FPATH)')"
    if [ -f "$DEFAULT_CHECKPOINT" ]; then
        CHECKPOINT="$DEFAULT_CHECKPOINT"
        print_warning "No checkpoint specified, using default: $CHECKPOINT"
    else
        print_error "No checkpoint specified and default checkpoint not found!"
        print_info "Please specify a checkpoint with --checkpoint or use --list to see available checkpoints"
        show_usage
        exit 1
    fi
fi

# Validate checkpoint file exists
if [ ! -f "$CHECKPOINT" ]; then
    print_error "Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Build the command
USE_LLM_FLAG=""
if [ "$USE_LLM" = true ]; then
    USE_LLM_FLAG="--use_llm"
fi

CMD="python \"$PYTHON_SCRIPT\" \
    --checkpoint_path \"$CHECKPOINT\" \
    --experiment \"$EXPERIMENT\" \
    --combo_id \"$COMBO_ID\" \
    --root_dir \"$ROOT_DIR\" \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --gpu_id $GPU_ID \
    $USE_LLM_FLAG \
    $FAST_DEV_RUN"

# Print configuration
print_info "Fine-tuning Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Experiment: $EXPERIMENT"
echo "  Combo ID: $COMBO_ID"
echo "  Root Dir: $ROOT_DIR"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Learning Rate: $LR"
echo "  GPU ID: $GPU_ID"
if [ "$USE_LLM" = true ]; then
    echo "  Using LLM dataset: YES"
else
    echo "  Using DIP dataset: YES"
fi
if [ -n "$FAST_DEV_RUN" ]; then
    echo "  Fast Dev Run: YES"
fi
echo

# Execute or show command
if [ "$DRY_RUN" = true ]; then
    print_info "Dry run - command that would be executed:"
    echo "$CMD"
else
    print_info "Starting fine-tuning..."
    eval $CMD
    if [ $? -eq 0 ]; then
        print_success "Fine-tuning completed successfully!"
    else
        print_error "Fine-tuning failed!"
        exit 1
    fi
fi
