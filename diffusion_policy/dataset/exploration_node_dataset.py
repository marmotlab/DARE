import sys
import os
# Add the parent directory of 'diffusion_policy' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


# HACK down size int64s to int16 and int32 to store
'''
Total Number of Time Steps: 25650
Total Number of Episodes: 206
Node Padding Size: 360
Edge Padding Size: 25
results/ground_truth/data.zarr
 ├── data
 │   ├── action (25650, 2) float32 #(x, y) distance moved
 │   ├── node_inputs (25650, 360, 5) float32
 │   ├── node_padding_mask (25650, 1, 360) int16
 │   ├── edge_mask (25650, 360, 360) int16 (upcast int64 when sampled for inference)
 │   ├── current_index (25650, 1, 1) int32 (upcast int64 when sampled for inference)
 │   ├── current_edge (25650, 25, 1) int32 (upcast int64 when sampled for inference)
 │   ├── edge_padding_mask (25650, 1, 25) int16
 └── meta
     └── episode_ends (206,) int32
Each array in data stores one data field from all episodes concatenated along the first dimension (time).
The meta/episode_ends array stores the end index for each episode along the fist dimension.
'''

class ExplorationNodeDataset(BaseImageDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        # self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['action',
        #                                                                   'node_inputs',
        #                                                                   'node_padding_mask',
        #                                                                   'edge_mask',
        #                                                                   'current_index',
        #                                                                   'current_edge',
        #                                                                   'edge_padding_mask'])
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        action = sample['action'].astype(np.float32)
        node_inputs = sample['node_inputs'].astype(np.float32)
        node_padding_mask = sample['node_padding_mask'].astype(np.int16)
        edge_mask = sample['edge_mask'].astype(np.int64)
        current_index = sample['current_index'].astype(np.int64)
        current_edge = sample['current_edge'].astype(np.int64)
        edge_padding_mask = sample['edge_padding_mask'].astype(np.int16)

        data = {
            'obs': {'node_inputs': node_inputs,
                    'node_padding_mask': node_padding_mask,
                    'edge_mask': edge_mask,
                    'current_index': current_index,
                    'current_edge': current_edge,
                    'edge_padding_mask': edge_padding_mask},
            'action': action
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/diffusion_exploration/results/ground_truth_train_4000_new/data.zarr')

    # parameters
    pred_horizon = 8
    obs_horizon = 2
    action_horizon = 1
    #|o|o|                             observations: 2
    #| |a|                             actions executed: 1
    #|p|p|p|p|p|p|p|p|                 actions predicted: 8


    dataset = ExplorationNodeDataset(zarr_path, horizon=pred_horizon, pad_before=obs_horizon-1, pad_after=action_horizon-1)
    normalizer = dataset.get_normalizer()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    from matplotlib import pyplot as plt
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    # img = dataset.replay_buffer['img'][75]
    # plt.imshow(img, cmap='gray')
    # plt.show()
    
    # visualize data in batch
    print("len(dataloader):", len(dataloader))
    for idx, batch in enumerate(dataloader):
        if idx == 0:
            print(batch.keys())

            # data = {
            #     'obs': {'node_inputs': node_inputs,
            #             'node_padding_mask': node_padding_mask,
            #             'edge_mask': edge_mask,
            #             'current_index': current_index,
            #             'current_edge': current_edge,
            #             'edge_padding_mask': edge_padding_mask},
            #     'action': action
            # }

            node_inputs = batch['obs']['node_inputs']
            node_padding_mask = batch['obs']['node_padding_mask']
            edge_mask = batch['obs']['edge_mask']
            current_index = batch['obs']['current_index']
            current_edge = batch['obs']['current_edge']
            edge_padding_mask = batch['obs']['edge_padding_mask']
            action = batch['action']

            print("batch['obs']['node_inputs'].shape:", batch['obs']['node_inputs'].shape)
            print("batch['obs']['node_padding_mask'].shape:", batch['obs']['node_padding_mask'].shape)
            print("batch['obs']['edge_mask'].shape:", batch['obs']['edge_mask'].shape)
            print("batch['obs']['current_index'].shape:", batch['obs']['current_index'].shape)
            print("batch['obs']['current_edge'].shape:", batch['obs']['current_edge'].shape)
            print("batch['obs']['edge_padding_mask'].shape:", batch['obs']['edge_padding_mask'].shape)
            print("batch['action'].shape", batch['action'].shape)

            print(f"min action: {action.min()}, max action: {action.max()}")
            naction = normalizer['action'].normalize(action)
            print("naction.shape:", naction.shape)
            print(f"min naction: {naction.min()}, max naction: {naction.max()}")

            for i, node_input in enumerate(batch['obs']['node_inputs'][0]):
                print(f"node_input{i}: {node_input}")
            for i, node_padding_mask in enumerate(batch['obs']['node_padding_mask'][0]):
                print(f"node_padding_mask{i}: {node_padding_mask}")
            for i, edge_mask in enumerate(batch['obs']['edge_mask'][0]):
                print(f"edge_mask{i}: {edge_mask}")
            for i, current_index in enumerate(batch['obs']['current_index'][0]):
                print(f"current_index{i}: {current_index}")
            for i, current_edge in enumerate(batch['obs']['current_edge'][0]):
                print(f"current_edge{i}: {current_edge}")
            for i, edge_padding_mask in enumerate(batch['obs']['edge_padding_mask'][0]):
                print(f"edge_padding_mask{i}: {edge_padding_mask}")
            for i, action in enumerate(batch['action'][0]):
                print(f"action{i}: {action}")

            break


if __name__ == '__main__':
    test()