from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.encoder.exploration_map_encoder import simple_resnet18, SimpleResNet 
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerMapDiscretePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            map_encoder: SimpleResNet,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            obs_encoder_group_norm=True,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        ## DISCRETE STUFF START
        self.num_class = 5 # HACK
        one_hot_action_dim = action_dim * self.num_class # 10
        self.one_hot_action_dim = one_hot_action_dim
        ## DISCRETE STUFF START

        # get feature dim
        map_feature_dim = map_encoder.output_channels
        state_feature_dim = shape_meta['obs']['agent_pos']['shape'][0]
        obs_feature_dim = map_feature_dim + state_feature_dim
                
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=map_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        # create diffusion model
        input_dim = one_hot_action_dim if obs_as_cond else (obs_feature_dim + one_hot_action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.map_encoder = map_encoder
        self.model = model
        self.parallel_map_encoder = nn.DataParallel(map_encoder)
        self.parallel_model = nn.DataParallel(model)

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=one_hot_action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
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
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
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
            model_output = model(trajectory, t, cond)

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
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.one_hot_action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            map_features = self.map_encoder(this_nobs['image']).reshape(B, To, -1)
            state_features = this_nobs['agent_pos'].reshape(B, To, -1)
            # (B*obs_horizon, obs_encoder_output_dim) to (B, obs_horizon, obs_encoder_output_dim)
            cond = torch.cat([map_features, state_features], dim=-1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            map_features = self.map_encoder(this_nobs['image']).reshape(B, To, -1)
            state_features = this_nobs['agent_pos'].reshape(B, To, -1)
            # reshape back to B, To, Do
            nobs_features = torch.cat([map_features, state_features], dim=-1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)

        ## DISCRETE STUFF START
        naction_pred = nsample[...,:Da] # (B, T, original_action_dim * num_class)
        # print(f"naction_pred.shape: {naction_pred.shape}")
        # print(f"naction_pred[0]: {naction_pred[0]}")
        action_pred = (naction_pred + 1) / 2 # HACK to undo the -1 to 1 normalisation
        # print(f"action_pred.shape: {action_pred.shape}")
        # print(f"action_pred[0]: {action_pred[0]}")
        # convert from noisy one hot to softmax probabilities to smooth it out and get the largest value
        action_pred = action_pred.view(B, T, self.action_dim, -1) # (B, T, action_dim, num_class)
        action_pred = F.softmax(action_pred, dim=-1) # (B, T, action_dim, num_class) 0 - 1
        # print(f"softmax action_pred.shape: {action_pred.shape}")
        # print(f"softmax action_pred[0]: {action_pred[0]}")
       
        # convert from softmax probabilities to action values
        index_to_value = {0: -8, 1: -4, 2: 0, 3: 4, 4: 8} # HACK
        indices = torch.argmax(action_pred, dim=-1)
        action_pred = torch.tensor([index_to_value[idx.item()] for idx in indices.view(-1)], device=indices.device)
        action_pred = action_pred.view(indices.shape)
        # print(f"action_pred.shape: {action_pred.shape}")
        # print(f"action_pred[0]: {action_pred[0]}")
        ## DISCRETE STUFF END

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
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

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.map_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # do not normalize action but convert to 10 dim one hot tensor
        actions = batch['action']
        batch_size = actions.shape[0]
        horizon = actions.shape[1]
        To = self.n_obs_steps

        ## DISCRETE STUFF START
        # convert action to one hot tensor
        # print(f"actions.shape: {actions.shape}")
        # print(f"actions[0]: {actions[0]}")
        value_to_index = {-8: 0, -4: 1, 0: 2, 4: 3, 8: 4} # HACK
        indices = torch.tensor([value_to_index[value.item()] for value in actions.view(-1)], device=actions.device)
        indices = indices.view(actions.shape)
        one_hot_action = F.one_hot(indices, num_classes=5).float()
        one_hot_action = one_hot_action.view(actions.shape[0], actions.shape[1], -1)
        # print(f"one_hot_action.shape: {one_hot_action.shape}")
        # print(f"one_hot_action[0]: {one_hot_action[0]}")
        # normalise one hot action
        none_hot_action = one_hot_action * 2 - 1
        # print(f"none_hot_action.shape: {none_hot_action.shape}")
        # print(f"none_hot_action[0]: {none_hot_action[0]}") 
        ## DISCRETE STUFF END

        # node inputs obs do not need to be normalized
        nobs = self.normalizer.normalize(batch['obs'])


        # handle different ways of passing observation
        cond = None
        trajectory = none_hot_action
        if self.obs_as_cond:
            # slice from (B, pred_horizon, obs_encoder_output_dim) to (B, obs_horizon, obs_encoder_output_dim)
            # reshape (B, obs_horizon, obs_encoder_output_dim) to (B*obs_horizon, obs_encoder_output_dim)
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            map_features = self.parallel_map_encoder(this_nobs['image']).reshape(batch_size, To, -1)
            state_features = this_nobs['agent_pos'].reshape(batch_size, To, -1)
            # (B*obs_horizon, obs_encoder_output_dim) to (B, obs_horizon, obs_encoder_output_dim)
            cond = torch.cat([map_features, state_features], dim=-1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = none_hot_action[:,start:end]
        else:
            # reshape (B, pred_horizon, ...) to (B*pred_horizon, ...)
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            map_features = self.parallel_map_encoder(this_nobs['image']).reshape(batch_size, horizon, -1)
            state_features = this_nobs['agent_pos'].reshape(batch_size, horizon, -1)
            # reshape back to (B, pred_horizon, obs_encoder_output_dim)
            nobs_features = torch.cat([map_features, state_features], dim=-1)
            trajectory = torch.cat([none_hot_action, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
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
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.parallel_model(noisy_trajectory, timesteps, cond)

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
