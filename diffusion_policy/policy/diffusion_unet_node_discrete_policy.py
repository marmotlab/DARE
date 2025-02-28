from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.encoder.exploration_node_encoder import ExplorationNodeEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetNodeDiscretePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: ExplorationNodeEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            training=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        ## DISCRETE STUFF START
        self.node_resolution = shape_meta['obs']['node_resolution']
        self.adjacent_neighbor = shape_meta['obs']['adjacent_neighbor']
        self.len_action_index = self.adjacent_neighbor * 2 + 1
        one_hot_action_dim = action_dim * self.len_action_index
        self.one_hot_action_dim = one_hot_action_dim

        values = [(i - self.adjacent_neighbor) * self.node_resolution for i in range(self.len_action_index)]
        self.value_to_index = {value: index for index, value in enumerate(values)}
        self.index_to_value = {index: value for index, value in enumerate(values)}
        print(f"value_to_index: {self.value_to_index}")
        print(f"index_to_value: {self.index_to_value}")
        ## DISCRETE STUFF START

        # get feature dim
        obs_feature_dim = shape_meta['obs']['embedding_dim']

        # create diffusion model
        input_dim = one_hot_action_dim + obs_feature_dim # if not conditioned per step
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = one_hot_action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.parallel_obs_encoder = nn.DataParallel(obs_encoder)
        self.parallel_model = nn.DataParallel(model)

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=one_hot_action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # node inputs obs do not need to be normalized
        obs = obs_dict
        value = next(iter(obs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.one_hot_action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_obs = dict_apply(obs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            obs_features = self.obs_encoder(this_obs['node_inputs'],
                                            this_obs['node_padding_mask'],
                                            this_obs['edge_mask'],
                                            this_obs['current_index'],
                                            this_obs['current_edge'],
                                            this_obs['edge_padding_mask'])
            # (B*obs_horizon, obs_encoder_output_dim) to (B, obs_horizon * obs_encoder_output_dim)
            global_cond = obs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_obs = dict_apply(obs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            obs_features = self.obs_encoder(this_obs['node_inputs'],
                                            this_obs['node_padding_mask'],
                                            this_obs['edge_mask'],
                                            this_obs['current_index'],
                                            this_obs['current_edge'],
                                            this_obs['edge_padding_mask'])
            # reshape back to (B, obs_horizon, obs_encoder_output_dim)
            obs_features = obs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = obs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        ## DISCRETE STUFF START
        naction_pred = nsample[...,:Da] # (B, T, original_action_dim * num_class)
        # print(f"naction_pred.shape: {naction_pred.shape}")
        # print(f"naction_pred[0]: {naction_pred[0]}")
        action_pred = (naction_pred + 1) / 2 # ussndo the -1 to 1 normalisation
        action_pred = action_pred.view(B, T, self.action_dim, -1) # (B, T, action_dim, num_class)
        # print(f"onehot action_pred.shape: {action_pred.shape}")
        # print(f"onehot action_pred[0]: {action_pred[0]}")
        # convert from onehot probabilities to action values
        indices = torch.argmax(action_pred, dim=-1)
        action_pred = torch.tensor([self.index_to_value[idx.item()] for idx in indices.view(-1)], device=indices.device)
        action_pred = action_pred.view(indices.shape)
        # print(f"action_pred.shape: {action_pred.shape}")
        # print(f"action_pred[0]: {action_pred[0]}")
        ## DISCRETE STUFF END
        
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        # do not normalize action but convert to 10 dim one hot tensor
        actions = batch['action']
        batch_size = actions.shape[0]
        horizon = actions.shape[1]
        
        ## DISCRETE STUFF START
        # convert action to one hot tensor
        # print(f"actions.shape: {actions.shape}")
        # print(f"actions[0]: {actions[0]}")
        indices = torch.tensor([self.value_to_index[value.item()] for value in actions.view(-1)], device=actions.device)
        indices = indices.view(actions.shape)
        one_hot_action = F.one_hot(indices, num_classes=self.len_action_index).float()
        one_hot_action = one_hot_action.view(actions.shape[0], actions.shape[1], -1)
        # print(f"one_hot_action.shape: {one_hot_action.shape}")
        # print(f"one_hot_action[0]: {one_hot_action[0]}")
        # normalise one hot action
        none_hot_action = one_hot_action * 2 - 1
        # print(f"none_hot_action.shape: {none_hot_action.shape}")
        # print(f"none_hot_action[0]: {none_hot_action[0]}") 
        ## DISCRETE STUFF END
        
        # node inputs obs do not need to be normalized
        obs = batch['obs']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = none_hot_action
        cond_data = trajectory
        if self.obs_as_global_cond:
            # slice from (B, pred_horizon, obs_encoder_output_dim) to (B, obs_horizon, obs_encoder_output_dim)
            # reshape (B, obs_horizon, obs_encoder_output_dim) to (B*obs_horizon, obs_encoder_output_dim)
            this_obs = dict_apply(obs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            obs_features = self.parallel_obs_encoder(this_obs['node_inputs'],
                                            this_obs['node_padding_mask'],
                                            this_obs['edge_mask'],
                                            this_obs['current_index'],
                                            this_obs['current_edge'],
                                            this_obs['edge_padding_mask'])
            # (B*obs_horizon, obs_encoder_output_dim) to (B, obs_horizon * obs_encoder_output_dim)
            global_cond = obs_features.reshape(batch_size, -1)
        else:
            # reshape (B, pred_horizon, ...) to (B*pred_horizon, ...)
            this_obs = dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))
            obs_features = self.parallel_obs_encoder(this_obs['node_inputs'],
                                            this_obs['node_padding_mask'],
                                            this_obs['edge_mask'],
                                            this_obs['current_index'],
                                            this_obs['current_edge'],
                                            this_obs['edge_padding_mask'])
            # reshape back to (B, pred_horizon, obs_encoder_output_dim)
            obs_features = obs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([none_hot_action, obs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.parallel_model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
