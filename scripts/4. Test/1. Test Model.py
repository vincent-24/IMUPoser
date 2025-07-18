# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

# Add project root to path to find constants.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule, get_dataset
from imuposer.utils import get_parser
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.models.LSTMs.IMUPoser_Model_FineTune import IMUPoserModelFineTune
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.smpl.parametricModel import ParametricModel
from constants import PROJECT_ROOT_DIR, BASE_MODEL_DIR, PROCESSED_TEST_DATA, TEST_SUBJECTS, TEST_RESULTS_DIR, TEST_CHECKPOINT

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()

combo_id = args.combo_id
_experiment = args.experiment
checkpoint_dir = args.checkpoint
output_root_dir = args.output_root_dir if args.output_root_dir else PROJECT_ROOT_DIR

if TEST_CHECKPOINT is BASE_MODEL_DIR:
    fine_tuned = False 
else:
    fine_tuned = True

checkpoint_file = ""
for fname in os.listdir(checkpoint_dir):
    if fname.endswith('.ckpt'):
        checkpoint_file = fname
    else:
        continue
    checkpoint_path = checkpoint_dir + checkpoint_file

    print(f"Testing model for combo_id: {combo_id}")
    print(f"Using checkpoint: {checkpoint_path}")

    # %%
    config = Config(
        experiment=f"{_experiment}_{combo_id}", 
        model="GlobalModelIMUPoser",
        project_root_dir=output_root_dir, 
        joints_set=amass_combos[combo_id], 
        normalize="no_translation",
        r6d=True, 
        loss_type="mse", 
        use_joint_loss=True, 
        device="0",
        test_dataset=True  # Use test dataset for testing
    )

    # %%
    # Load the trained model from checkpoint
    print("Loading model from checkpoint...")
    print(f"Model type: {'Fine-tuned' if fine_tuned else 'General'}")
    print(f"Checkpoint file exists: {pathlib.Path(checkpoint_path).exists()}")
    print(f"Checkpoint size: {pathlib.Path(checkpoint_path).stat().st_size if pathlib.Path(checkpoint_path).exists() else 'N/A'} bytes")

    if fine_tuned:
        # Load fine-tuned model
        dummy_model = IMUPoserModel(config)
        model = IMUPoserModelFineTune.load_from_checkpoint(
            checkpoint_path,
            config=config,
            pretrained_model=dummy_model
        )
    else:
        # Load general model
        model = IMUPoserModel.load_from_checkpoint(
            checkpoint_path,
            config=config
        )
    
    model.eval()  # Set to evaluation mode

    # Print some model parameters to verify it loaded correctly
    print(f"Model device: {next(model.parameters()).device}")
    print(f"First few model parameters: {list(model.parameters())[0].flatten()[:5]}")  # Show first 5 parameters

    # Load only the test dataset
    print("Loading test dataset...")
    test_dataset = get_dataset(config, test_only=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        collate_fn=lambda batch: (
            torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True),
            torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True),
            [item[0].shape[0] for item in batch],
            [item[1].shape[0] for item in batch]
        ), 
        num_workers=8, 
        shuffle=False
    )

    # Determine test subjects from the processed data
    test_data_path = config.processed_imu_poser_25fps / PROCESSED_TEST_DATA
    print(f"Loading test data from: {test_data_path}")

    # Determine which subjects were actually used in the test split by reading the preprocessing script logic
    # We need to match the exact logic from process_dipimu(split="test") in the preprocessing script
    raw_data_path = config.raw_llm_path
    test_subjects = []

    # Read the current test split definition from preprocessing (this should match 1. preprocess_all.py)
    # Based on the current preprocessing script, test split is ['s_11', 's_12']
    if raw_data_path.exists():
        # Check which of the expected test subjects actually exist in the raw data
        expected_test_subjects = TEST_SUBJECTS  # This should match the preprocessing script
        all_subjects = [d.name for d in raw_data_path.iterdir() if d.is_dir() and d.name.startswith('s_')]
        test_subjects = [s for s in expected_test_subjects if s in all_subjects]

    if not test_subjects:
        test_subjects = TEST_SUBJECTS  # fallback to match preprocessing script

    test_subjects_str = ", ".join(test_subjects)
    print(f"Test dataset size: {len(test_dataset)} sequences")
    print(f"Test subjects: {test_subjects_str}")

    # %%
    # Initialize evaluation metrics
    device = model.device
    all_losses = []
    all_pose_losses = []
    all_joint_losses = []
    all_predictions = []
    all_targets = []

    # Initialize SMPL body model for joint position calculation if needed
    if config.use_joint_loss:
        bodymodel = ParametricModel(config.og_smpl_model_path, device=device)

    print("Starting evaluation on test set...")

    # %%
    # Evaluate the model
    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            imu_inputs, target_pose, input_lengths, _ = batch
            imu_inputs = imu_inputs.to(device)
            target_pose = target_pose.to(device)
            
            # Forward pass
            _pred = model(imu_inputs, input_lengths)
            
            # Handle different model output formats
            if fine_tuned:
                # Fine-tuned models return predictions directly
                pred_pose = _pred[:, :, :model.n_pose_output]
            else:
                # General models may return predictions directly or in a different format
                if isinstance(_pred, tuple):
                    pred_pose = _pred[0][:, :, :model.n_pose_output]
                else:
                    pred_pose = _pred[:, :, :model.n_pose_output]
            
            target_pose_truncated = target_pose[:, :, :model.n_pose_output]
            
            # Calculate pose loss (MSE between predicted and ground truth poses)
            pose_loss = torch.nn.functional.mse_loss(pred_pose, target_pose_truncated)
            total_loss = pose_loss
            
            # Calculate joint position loss if enabled
            joint_loss = torch.tensor(0.0).to(device)
            if config.use_joint_loss:
                pred_joint = bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
                target_joint = bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose_truncated).view(-1, 216))[1]
                joint_loss = torch.nn.functional.mse_loss(pred_joint, target_joint)
                total_loss += joint_loss
            
            # Store metrics
            all_losses.append(total_loss.item())
            all_pose_losses.append(pose_loss.item())
            all_joint_losses.append(joint_loss.item())
            
            # Store predictions and targets for additional analysis
            all_predictions.append(pred_pose.cpu())
            all_targets.append(target_pose_truncated.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_dataloader)}")

    # %%
    # Calculate final metrics
    avg_total_loss = np.mean(all_losses)
    avg_pose_loss = np.mean(all_pose_losses)
    avg_joint_loss = np.mean(all_joint_losses) if config.use_joint_loss else 0.0

    print("\n" + "="*60)
    print(f"TEST RESULTS ON DIP-IMU TEST SET ({test_subjects_str})")
    print(f"Model Type: {'Fine-tuned' if fine_tuned else 'General'}")
    print("="*60)
    print(f"Average Total Loss: {avg_total_loss:.6f}")
    print(f"Average Pose Loss (MSE): {avg_pose_loss:.6f}")
    if config.use_joint_loss:
        print(f"Average Joint Position Loss (MSE): {avg_joint_loss:.6f}")
    print(f"Total test sequences: {len(test_dataset)}")
    print(f"Combo ID: {combo_id}")
    print(f"Joints set: {config.joints_set}")
    print("="*60)

    # %%
    # Calculate additional metrics for analysis
    print("\nADDITIONAL METRICS:")
    print("-" * 30)

    # Calculate per-joint metrics if using joint loss
    if config.use_joint_loss:
        all_pred_joints = []
        all_target_joints = []
        
        with torch.no_grad():
            for pred_batch, target_batch in zip(all_predictions, all_targets):
                pred_batch = pred_batch.to(device)
                target_batch = target_batch.to(device)
                
                pred_joints = bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_batch).view(-1, 216))[1]
                target_joints = bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_batch).view(-1, 216))[1]
                
                all_pred_joints.append(pred_joints.cpu())
                all_target_joints.append(target_joints.cpu())
        
        # Calculate MPJPE (Mean Per Joint Position Error) in millimeters
        all_pred_joints = torch.cat(all_pred_joints, dim=0)  # [N, 24, 3]
        all_target_joints = torch.cat(all_target_joints, dim=0)  # [N, 24, 3]
        
        joint_errors = torch.norm(all_pred_joints - all_target_joints, dim=2)  # [N, 24]
        mpjpe_per_joint = torch.mean(joint_errors, dim=0) * 1000  # Convert to mm
        mpjpe_overall = torch.mean(mpjpe_per_joint)
        
        print(f"MPJPE (Mean Per Joint Position Error): {mpjpe_overall:.2f} mm")
        print(f"MPJPE std: {torch.std(joint_errors.flatten()) * 1000:.2f} mm")
        
        # Print per-joint errors
        joint_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
                    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
                    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
        
        print("\nPer-joint MPJPE (mm):")
        for i, (joint_name, error) in enumerate(zip(joint_names, mpjpe_per_joint)):
            print(f"  {i:2d} {joint_name:15s}: {error:.2f}")

    # Calculate pose error statistics
    all_pred_poses = torch.cat(all_predictions, dim=0)
    all_target_poses = torch.cat(all_targets, dim=0)
    pose_errors = torch.norm(all_pred_poses - all_target_poses, dim=2)
    print(f"\nPose Error Statistics:")
    print(f"  Mean: {torch.mean(pose_errors):.6f}")
    print(f"  Std:  {torch.std(pose_errors):.6f}")
    print(f"  Min:  {torch.min(pose_errors):.6f}")
    print(f"  Max:  {torch.max(pose_errors):.6f}")

    print("\nEvaluation completed!")

    # %%
    # Save results to file with descriptive directory name
    import datetime
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    test_subjects_filename = "_".join(test_subjects)  # e.g., "s09_s10"
    model_type = "finetune" if fine_tuned else "general"
    results_dir_name = f"test_{model_type}_{combo_id}_{test_subjects_filename}-{date_str}"
    results_dir = os.path.join(TEST_RESULTS_DIR, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    results_filename = f"test_results_{checkpoint_file.split('.')[0]}.txt"
    results_file = os.path.join(results_dir, results_filename)

    with open(results_file, "w") as f:
        f.write(f"TEST RESULTS ON DIP-IMU TEST SET ({test_subjects_str})\n")
        f.write(f"Model Type: {'Fine-tuned' if fine_tuned else 'General'}\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Combo ID: {combo_id}\n")
        f.write(f"Joints set: {config.joints_set}\n")
        f.write(f"Test subjects: {test_subjects_str}\n")
        f.write(f"Total test sequences: {len(test_dataset)}\n")
        f.write(f"Average Total Loss: {avg_total_loss:.6f}\n")
        f.write(f"Average Pose Loss (MSE): {avg_pose_loss:.6f}\n")
        if config.use_joint_loss:
            f.write(f"Average Joint Position Loss (MSE): {avg_joint_loss:.6f}\n")
            f.write(f"MPJPE (Mean Per Joint Position Error): {mpjpe_overall:.2f} mm\n")
            f.write(f"MPJPE std: {torch.std(joint_errors.flatten()) * 1000:.2f} mm\n")
            
            # Add detailed per-joint metrics
            f.write("\nPer-joint MPJPE (mm):\n")
            joint_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
                        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
                        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
                        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
            for i, (joint_name, error) in enumerate(zip(joint_names, mpjpe_per_joint)):
                f.write(f"  {i:2d} {joint_name:15s}: {error:.2f}\n")
        
        # Add pose error statistics
        f.write(f"\nPose Error Statistics:\n")
        f.write(f"  Mean: {torch.mean(pose_errors):.6f}\n")
        f.write(f"  Std:  {torch.std(pose_errors):.6f}\n")
        f.write(f"  Min:  {torch.min(pose_errors):.6f}\n")
        f.write(f"  Max:  {torch.max(pose_errors):.6f}\n")
        f.write("="*60 + "\n")

    print(f"Results saved to: {results_file}")
    print(f"Results directory: {results_dir}")
    print("Test evaluation completed successfully!\n\n")