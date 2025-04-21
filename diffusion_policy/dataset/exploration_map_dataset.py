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


'''
TODO obs horizon etc.. and check if padding works (padding works etc, further process for action and obs based on To n Ta)
TODO normaliser?
data/exploration_dataset.zarr
 ├── data
 │   ├── action (25650, 2) float32 #(x, y) distance moved
 │   ├── img (25650, 250, 250) float32 #map grayscale 0-255 image
 │   └── state (25650, 2) float32 #(x, y) position
 └── meta
     └── episode_ends (206,) int64
Each array in data stores one data field from all episodes concatenated along the first dimension (time).
The meta/episode_ends array stores the end index for each episode along the fist dimension.
'''

class ExplorationMapDataset(BaseImageDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        # self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['img', 'state', 'action']) # save data as .np
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
        data = {'action': self.replay_buffer['action'], 'agent_pos': self.replay_buffer['state'][...,:]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:].astype(np.float32)
        image = sample['img'].astype(np.float32)/255
        if len(image.shape) == 3: # add channel dimension
            image = np.expand_dims(image, axis=1)
        data = {'obs': {'image': image, 'agent_pos': agent_pos}, 'action': sample['action'].astype(np.float32)}
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/diffusion_exploration/results/ground_truth_image_train_1/data.zarr')

    # parameters
    pred_horizon = 8
    obs_horizon = 2
    action_horizon = 1
    #|o|o|                             observations: 2
    #| |a|                             actions executed: 1
    #|p|p|p|p|p|p|p|p|                 actions predicted: 8


    dataset = ExplorationMapDataset(zarr_path, horizon=pred_horizon, pad_before=obs_horizon-1, pad_after=action_horizon-1)
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
        if idx == 74:
            print(f"Batch {idx}")
            print(batch.keys())

            image = batch['obs']['image']
            agent_pos = batch['obs']['agent_pos']
            action = batch['action']
            print("batch['obs']['image'].shape:", batch['obs']['image'].shape)
            print("batch['obs']['agent_pos'].shape:", batch['obs']['agent_pos'].shape)
            print("batch['action'].shape", batch['action'].shape)

            unique_values = np.unique(image)
            print("Unique values in image:", unique_values)
            print(f"min image: {image.min()}, max image: {image.max()}")
            print(f"min agent_pos: {agent_pos.min()}, max agent_pos: {agent_pos.max()}")
            print(f"min action: {action.min()}, max action: {action.max()}")

            nimage = normalizer['image'].normalize(image)
            nagent_pos = normalizer['agent_pos'].normalize(agent_pos)
            naction = normalizer['action'].normalize(action)
            print("nimage.shape:", nimage.shape)
            print("nagent_pos.shape:", nagent_pos.shape)
            print("naction.shape:", naction.shape)

            unique_values = np.unique(nimage.detach().numpy())
            print("Unique values in nimage:", unique_values)
            print(f"min nimage: {nimage.min()}, max nimage: {nimage.max()}")
            print(f"min nagent_pos: {nagent_pos.min()}, max nagent_pos: {nagent_pos.max()}")
            print(f"min naction: {naction.min()}, max naction: {naction.max()}")

            for image in batch['obs']['image'][0]:
                unique_values = np.unique(image)
                plt.imshow(image.permute(1, 2, 0), cmap='gray')
                plt.show()

            for i, agent_pos in enumerate(batch['obs']['agent_pos'][0]):
                print(f"agent_pos{i}: {agent_pos}")

            for i, action in enumerate(batch['action'][0]):
                print(f"action{i}: {action}")

            # break


if __name__ == '__main__':
    test()