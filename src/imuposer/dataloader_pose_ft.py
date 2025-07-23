import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from .config import (
    smpl_vertices_corresponding_joints, 
    body_model,
    dip_imu_sensor_vertex_ids,
    xsens_imu_sensor_vertex_ids,
    total_capture_sensor_vertex_ids
)  

#
# # dataset_path = "/Volumes/harddrive/dataset/pose/work/"
# dataset_path = "/home/ubuntu/imucoco/dataset/work"
# # device = torch.device('cuda:0')


def collate_fn(batch):
    # Initialize dictionaries to store batched data
    batched_data = {}
    seq_attribute_names = [
        'vimu_joints', 'imu', 'joint_velocity', 'joint_position', 'joint_orientation', 'pose_local', 'pose_global', 'tran', 'ft_contact'
    ]
    for attribute_name in seq_attribute_names:
        if any(attribute_name in sample for sample in batch):
            batched_data[attribute_name] = torch.stack([sample[attribute_name] for sample in batch])

    if any('imu_corresponding_vertices' in sample for sample in batch):
        batched_data['imu_corresponding_vertices'] = torch.stack([sample['imu_corresponding_vertices'] for sample in batch if 'imu_corresponding_vertices' in sample])

    if any('velocity_init' in sample for sample in batch):
        batched_data['velocity_init'] = torch.stack(
            [sample['velocity_init'] for sample in batch if 'velocity_init' in sample]
        )
    if any('position_init' in sample for sample in batch):
        batched_data['position_init'] = torch.stack(
            [sample['position_init'] for sample in batch if 'position_init' in sample]
        )
    if any('tran_mask' in sample for sample in batch):
        batched_data['tran_mask'] = torch.stack([sample['tran_mask'] for sample in batch if 'tran_mask' in sample])

    if any('activity_name' in sample for sample in batch):
        batched_data['activity_name'] = [sample['activity_name'] for sample in batch]

    if any('file_name' in sample for sample in batch):
        batched_data['file_name'] = [sample['file_name'] for sample in batch]

    if any('dominant_hand' in sample for sample in batch):
        batched_data['dominant_hand'] = [sample['dominant_hand'] for sample in batch]


    batched_data['sequence_lengths'] = torch.tensor([sample['sequence_lengths'] for sample in batch]).long()

    return batched_data

class PoseDataset(Dataset):
    def __init__(self,
                 dataset_path="/home/ubuntu/imucoco/dataset/work",
                 datasets=None,
                 split='train',
                 device='cuda:0',
                 parse_vjoints=True,
                 parse_imu=True,
                 parse_local_pose=True,
                 parse_global_pose=True,
                 parse_contact=True,
                 parse_tran=True,
                 parse_joint_vel=True,
                 parse_joint_pos=True,
                 parse_joint_ori=True,
                 parse_vinit=True,
                 parse_pinit=True,
                 use_joint_asp=True,
                 joint_attr_to_root=True,
                 local_pose_r6d=True,
                 global_pose_r6d=True,
                 parse_activity_name=False,
                 parse_file_name=False,
                 parse_dominant_hand=False,
                 sequence_length=300
                 ):
        super(PoseDataset, self).__init__()

        if datasets is None:
            datasets = ['DIP_IMU_train', 'XSens', 'AMASS']

        self.dataset_path = dataset_path
        self.datasets = datasets
        self.split = split
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.sequence_length = sequence_length

        self.parse_vjoints = parse_vjoints
        self.parse_imu = parse_imu
        self.parse_local_pose = parse_local_pose
        self.parse_global_pose = parse_global_pose
        self.parse_tran = parse_tran
        self.parse_contact = parse_contact
        self.parse_activity_name = parse_activity_name
        self.parse_file_name = parse_file_name
        self.parse_dominant_hand = parse_dominant_hand

        self.parse_joint_vel = parse_joint_vel
        self.parse_joint_pos = parse_joint_pos
        self.parse_joint_ori = parse_joint_ori

        self.parse_vinit = parse_vinit
        self.parse_pinit = parse_pinit

        self.local_pose_r6d = local_pose_r6d
        self.global_pose_r6d = global_pose_r6d

        self.use_joint_asp = use_joint_asp
        self.joint_attr_to_root = joint_attr_to_root

        self.dipimu_imu_sensor_vertices = torch.tensor(list(dip_imu_sensor_vertex_ids.values()))
        self.xsens_imu_sensor_vertices = torch.tensor(list(xsens_imu_sensor_vertex_ids.values()))
        self.totalcapture_imu_sensor_vertices = (-1) * torch.ones(17)
        self.totalcapture_imu_sensor_vertices[:len(total_capture_sensor_vertex_ids.values())] = torch.tensor(list(total_capture_sensor_vertex_ids.values()))
        self.amass_imu_sensor_vertices = (-1) * torch.ones(17)

        self.samples = []
        self.samples_energy = []

        self.prepare_data()
        print("Number of samples: ", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def prepare_data(self):
        if self.split == 'train':
            # then use the train meta csv to load the samples
            for dataset in self.datasets:
                dataset_meta_file = os.path.join(self.dataset_path, f"{dataset}.csv")
                dataset_meta_df = pd.read_csv(dataset_meta_file)
                for _, row in dataset_meta_df.iterrows():
                    sample = {}
                    sample['dataset_name'] = dataset
                    sample['file_name'] = os.path.join(self.dataset_path, dataset, row['file_name'])
                    print("sample['file_name']", sample['file_name'])
                    sample['length'] = row['length']
                    sample['kinematic_energy'] = row['kinematic_energy']

                    if 'XSens' in dataset:
                        sample['kinematic_energy'] = sample['kinematic_energy'] * 10

                    if 'DIP' in dataset:
                        sample['kinematic_energy'] = sample['kinematic_energy'] * 3

                    self.samples.append(sample)
                    self.samples_energy.append(row['kinematic_energy'])
                print(f"Number of samples in {dataset}: {len(dataset_meta_df)}")
            self.samples_energy = np.asarray(self.samples_energy)
        else:
            # if it is test set, just list the files
            for dataset in self.datasets:
                dataset_dir = os.path.join(self.dataset_path, dataset)
                print("listing files in ", dataset_dir + '/*.pt')
                for file_name in sorted(glob.glob(dataset_dir + '/*.pt')):
                    print(file_name, )
                    sample = {}
                    sample['dataset_name'] = dataset
                    sample['file_name'] = file_name
                    self.samples.append(sample)
            print(f"Number of testing samples {len(self.samples)}")

    def __getitem__(self, index):
        sample = self.samples[index]
        dataset_name = sample['dataset_name']

        # Load the original data (a helper function that retrieves the data structure)
        start_time_load = datetime.now()
        # print("Loading data")
        out_data = torch.load(sample['file_name'])
        end_time = datetime.now()
        # print(f"Loading time: {end_time - start_time_load}")

        # Extract the segmentation range

        # Dictionary to store parsed tensors based on flags
        parsed_data = {}
        seq_attribute_names = []

        if self.parse_vjoints:
            parsed_data['vimu_joints'] = out_data['vimu']['vimu_joints']
            seq_attribute_names.append('vimu_joints')
        if self.parse_imu:
            if dataset_name == 'DIP_IMU_train_real_imu_position_only' or dataset_name == 'DIP_IMU_train_real_imu_position_only_asp':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.dipimu_imu_sensor_vertices
            elif dataset_name == 'XSens_real_imu_position_only' or dataset_name == 'XSens_real_imu_position_only_asp':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.xsens_imu_sensor_vertices
            elif dataset_name == 'TotalCapture_real_imu_position_only':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.totalcapture_imu_sensor_vertices
            elif dataset_name == 'AMASS_real_imu_position_only' or dataset_name == 'AMASS_real_imu_position_only_asp':
                # for fine-tuning poser, directly use amass's vimu_mesh, check data_augment.py
                parsed_data['imu'] = out_data['vimu']['vimu_mesh'].float()
                parsed_data['imu_corresponding_vertices'] = self.dipimu_imu_sensor_vertices
            else:
                parsed_data['imu'] = out_data['imu']['imu'].float()
            seq_attribute_names.append('imu')

        if self.parse_local_pose:
            if self.local_pose_r6d:
                parsed_data['pose_local'] = out_data['gt']['pose_local'][:, :, :, :2].transpose(2, 3).flatten(2).float()
            else:
                parsed_data['pose_local'] = out_data['gt']['pose_local'].float()
            seq_attribute_names.append('pose_local')
        if self.parse_global_pose:
            if self.global_pose_r6d:
                parsed_data['pose_global'] = out_data['joint']['orientation'][:, :, :, :2].transpose(2, 3).clone().flatten(2)  # global pose is just the global joint orientation
            else:
                parsed_data['pose_global'] = out_data['joint']['orientation']
            seq_attribute_names.append('pose_global')
        if self.parse_tran:
            if out_data['gt']['tran'] is not None:
                parsed_data['tran_mask'] = torch.tensor(1).bool()
                parsed_data['tran'] = out_data['gt']['tran'].float()
                # print("parsed_data['tran']1", parsed_data['tran'] )
            else:
                parsed_data['tran_mask'] = torch.tensor(0).bool()
                parsed_data['tran'] = torch.zeros(out_data['gt']['pose_local'].shape[0], 3).float()
                # print("parsed_data['tran']2", parsed_data['tran'])
            seq_attribute_names.append('tran')
        if self.parse_contact:
            parsed_data['ft_contact'] = out_data['gt']['ft_contact'].float()
            seq_attribute_names.append('ft_contact')

        if self.parse_joint_vel or self.parse_vinit:
            if self.use_joint_asp:
                parsed_data['joint_velocity'] = out_data['joint']['asp_velocity']
                # raise NotImplementedError
            else:
                parsed_data['joint_velocity'] = out_data['joint']['velocity']
                # root_vel = torch.cat((torch.zeros(1, 3), parsed_data['tran'][1:] - parsed_data['tran'][:-1])) * 60
                # # print("root velocity 1: ", parsed_data['joint_velocity'][:, 0], parsed_data['joint_velocity'][:, 0].shape)
                # # print("root velocity tran: ", root_vel, root_vel.shape)
                # parsed_data['joint_velocity'][:, 0] = root_vel
                # root_int_tran = torch.stack([root_vel[:i+1].sum(dim=0) for i in range(root_vel.shape[0])]) / 60
                # print("root integration tran", root_int_tran)
            if self.joint_attr_to_root:
                raise NotImplementedError
                # parsed_data['joint_velocity'][:, 1:] = (parsed_data['joint_velocity'][:, 1:] - parsed_data['joint_velocity'][:, :1]).bmm(out_data['joint']['orientation'][:, 0])
            seq_attribute_names.append('joint_velocity')

        if self.parse_joint_pos:
            # _, gt_joints_positions = body_model.forward_kinematics(pose=out_data['gt']['pose_local'].float(), calc_mesh=False)
            if self.use_joint_asp:
                parsed_data['joint_position'] = out_data['joint']['asp_position']
            else:
                parsed_data['joint_position'] = out_data['joint']['position']
            #     raise NotImplementedError
            # else:
            #     parsed_data['joint_position'] = gt_joints_positions
            if self.joint_attr_to_root:
                raise NotImplementedError

            parsed_data['joint_position'][:, 1:] = (parsed_data['joint_position'][:, 1:] - parsed_data['joint_position'][:, :1])
            seq_attribute_names.append('joint_position')

        if self.parse_joint_ori:
            # this is global joint orientation
            parsed_data['joint_orientation'] = out_data['joint']['orientation']
            parsed_data['joint_orientation'] = parsed_data['joint_orientation'][:, :, :, :2].transpose(2, 3).clone().flatten(2)
            seq_attribute_names.append('joint_orientation')

        if self.parse_vinit:
            parsed_data['velocity_init'] = parsed_data['joint_velocity'][0]  # Adjust if needed
        if self.parse_pinit:
            parsed_data['position_init'] = parsed_data['joint_position'][0]  # Adjust if needed

        if self.parse_activity_name:
            parsed_data['activity_name'] = out_data['gt']['activity_name']

        if self.parse_file_name:
            parsed_data['file_name'] = os.path.basename(sample['file_name'])

        if self.parse_dominant_hand:
            parsed_data['dominant_hand'] = out_data['bio']['dominant']
            print("parsed_data['dominant_hand']",  parsed_data['dominant_hand'])

        # seq_attribute_names = [
        #     'vimu_joints', 'imu', 'joint_velocity', 'joint_position', 'joint_orientation', 'pose_local', 'pose_global', 'tran', 'ft_contact'
        # ]

        parsed_data['sequence_lengths'] = out_data['gt']['pose_local'].shape[0]
        # TODO, check if put the length here has any influence
        for attr in seq_attribute_names:
            if attr in parsed_data:
                seq = parsed_data[attr]
                # print("attr", attr)
                seq_length = seq.shape[0]
                if seq_length < self.sequence_length:
                    padding = torch.zeros((self.sequence_length - seq_length, *seq.shape[1:]))
                    parsed_data[attr] = torch.cat((seq, padding), dim=0)
        return {key: torch.nn.utils.rnn.pad_sequence(value, batch_first=True) if key in seq_attribute_names else value for key, value in parsed_data.items()}


def get_ft_pose_dataloader(dataset_path="/home/ubuntu/imucoco/dataset/work",
                           datasets=['DIP_IMU_real_imu_position_only', 'XSens_real_imu_position_only', 'AMASS_real_imu_position_only', 'TotalCapture_real_imu_position_only'],
                           seq_len=300, device='cuda:0', batch_size=32,
                           parse_vjoints=True, parse_imu=True,
                           parse_local_pose=True, parse_global_pose=True,
                           parse_joint_vel=True, parse_joint_pos=True,
                           parse_joint_ori=True,
                           parse_contact=True,
                           parse_tran=True, parse_vinit=True, parse_pinit=True,
                           parse_activity_name=False, parse_file_name=False, parse_dominant_hand=False,
                           use_joint_asp=True, joint_attr_to_root=True,
                           use_kinematic_energy_sampling=False,
                           use_kinematic_energy_sampling_steps_per_epoch=200,
                           val_split=0.1, is_test_set=False,
                           workers=0,
                           prefetch_factor=None,
                           local_pose_r6d=True,
                           global_pose_r6d=True,
                           ):
    if is_test_set:
        dataset = PoseDataset(dataset_path=dataset_path,
                              datasets=datasets, device=device,
                              parse_vjoints=parse_vjoints,
                              parse_imu=parse_imu,
                              parse_local_pose=parse_local_pose,
                              parse_global_pose=parse_global_pose,
                              parse_joint_ori=parse_joint_ori,
                              parse_tran=parse_tran,
                              parse_joint_vel=parse_joint_vel,
                              parse_joint_pos=parse_joint_pos,
                              parse_vinit=parse_vinit,
                              parse_pinit=parse_pinit,
                              parse_contact=parse_contact,
                              parse_activity_name=parse_activity_name,
                              parse_file_name=parse_file_name,
                              parse_dominant_hand=parse_dominant_hand,
                              use_joint_asp=use_joint_asp,
                              split='test',
                              joint_attr_to_root=joint_attr_to_root,
                              sequence_length=seq_len,
                              local_pose_r6d=local_pose_r6d,
                              global_pose_r6d=global_pose_r6d
                              )
        test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
        return test_data_loader

    else:
        dataset = PoseDataset(dataset_path=dataset_path,
                              datasets=datasets, device=device,
                              parse_vjoints=parse_vjoints,
                              parse_imu=parse_imu,
                              parse_local_pose=parse_local_pose,
                              parse_global_pose=parse_global_pose,
                              parse_tran=parse_tran,
                              parse_joint_vel=parse_joint_vel,
                              parse_joint_pos=parse_joint_pos,
                              parse_joint_ori=parse_joint_ori,
                              parse_vinit=parse_vinit,
                              parse_pinit=parse_pinit,
                              parse_contact=parse_contact,
                              parse_activity_name=parse_activity_name,
                              parse_file_name=parse_file_name,
                              parse_dominant_hand=parse_dominant_hand,
                              use_joint_asp=use_joint_asp,
                              split='train',
                              joint_attr_to_root=joint_attr_to_root,
                              sequence_length=seq_len,
                              local_pose_r6d=local_pose_r6d,
                              global_pose_r6d=global_pose_r6d
                              )

        if val_split > 0:
            # split the dataset into training and validation sets
            train_size = int((1 - val_split) * len(dataset))
            indices = torch.randperm(len(dataset)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            if use_kinematic_energy_sampling:
                train_energy_weights = dataset.samples_energy[train_indices]
                # TODO: change the replacement to False seems to increase training efficiency by a lot 03/03/2025
                sampler_train = WeightedRandomSampler(train_energy_weights, num_samples=use_kinematic_energy_sampling_steps_per_epoch * batch_size, replacement=False)
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
            else:
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)

            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)

            return train_data_loader, val_data_loader
        else:
            if use_kinematic_energy_sampling:
                train_energy_weights = dataset.samples_energy
                sampler_train = WeightedRandomSampler(train_energy_weights, num_samples=use_kinematic_energy_sampling_steps_per_epoch * batch_size, replacement=False)
                train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
                return train_data_loader
            else:
                train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
                return train_data_loader
