# -----------------------------------------------------------------------------
# PPO (Proximal Policy Optimization) — High-Quality Single-File Implementation
# Author: Abhinav Bhatia
# Source: https://gist.github.com/bhatiaabhinav/edb07949471c0ae9e71811146cd46311
#
# Description:
#   A high-quality, truly single-file implementation of PPO (Proximal Policy Optimization),
#   designed for transparency, simplicity, and research-grade reproducibility.
#
#   Key features:
#     • Supports both continuous and discrete action spaces.
#     • Compatible with feedforward and recurrent (RNN-based) policies.
#     • Includes a Lagrange penalty-based constrained-MDP solver.
#     • No external dependencies beyond `torch` and `gymnasium`.
#     • Compact, well-structured, and easy to extend for experiments.
#     • Ideal for benchmarking, academic study, or educational use.
#
# License:
#   Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
#   You are free to share and adapt this work, provided appropriate credit is given.
#   See: https://creativecommons.org/licenses/by/4.0/
#
# Citation:
#   If you use or build upon this implementation in research or publications,
#   please consider citing the following:
#
#   @misc{bhatia2025ppo,
#     author       = {Abhinav Bhatia},
#     title        = {PPO (Proximal Policy Optimization) — High-Quality Single-File Implementation},
#     year         = {2025},
#     howpublished = {\url{https://gist.github.com/bhatiaabhinav/edb07949471c0ae9e71811146cd46311}}
#   }
#
# -----------------------------------------------------------------------------


import abc
import copy
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium import Env, make, wrappers
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from torch import Tensor, nn

autoreset_kwarg_samestep_newgymapi = {}
try:
    from gymnasium.vector import AutoresetMode
    autoreset_kwarg_samestep_newgymapi['autoreset_mode'] = AutoresetMode.SAME_STEP
except ImportError:  # older gymnasium version reset logic is already SAME_STEP by default
    pass


# --- Utility functions ---


def log_normal_prob(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Compute log probability of x under a normal distribution with given mean and std."""
    var = std ** 2
    log_prob = -0.5 * (((x - mean) ** 2) / var + 2 * torch.log(std + 1e-6) + np.log(2 * np.pi))
    return log_prob.sum(axis=-1, keepdim=True)


def gaussian_entropy(logstd: Tensor) -> Tensor:
    """Compute the entropy of a Gaussian distribution given its log standard deviation."""
    entropy = torch.sum(logstd + 0.5 * (1.0 + np.log(2 * np.pi)), dim=-1, keepdim=True)  # (batch_size, 1) or (batch_size, seq_len, 1)
    return entropy


def normalize_and_update_stats(mean: Tensor, var: Tensor, count: Tensor, x: Tensor, update_stats: bool):
    """Normalize input tensor x using running mean/var stats, optionally updating the stats in-place."""
    if update_stats:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * (count * batch_count / tot_count)
        new_var = m_2 / tot_count
        new_count = tot_count
        mean.data.copy_(new_mean)
        var.data.copy_(new_var)
        count.data.copy_(new_count)
    x = (x - mean) / torch.sqrt(var + 1e-8)
    x = torch.clamp(x, -10.0, 10.0)  # avoid extreme values
    return x


class SequenceModel(nn.Module):
    """
    A model that may contain some unidirectional `nn.RNNBase` layers. This class helps with context management of the RNN hidden states. On forward, it accepts an optional `cache` argument which is a list of hidden states for the RNN layers in the model. It returns the output and the updated cache. If no cache is provided, it initializes the hidden states to zeros.
    """

    def __init__(self, model):
        super(SequenceModel, self).__init__()
        self.model = model

    def forward(self, x: Tensor, cache: Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]) -> Tuple[Tensor, Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]]:
        """Forward pass through the model, managing RNN hidden states if present. cache[i] is the hidden state for the i-th RNN layer. It may be of any format depending on the RNN layer type. For example, for nn.LSTM, it is a tuple of (h, c). Assumes the input x is of shape (batch_size, seq_len, feature_dims...)"""
        assert len(x.shape) >= 3, "Input x must be of shape (batch_size, seq_len, feature_dims...)"
        if cache is not None:
            assert isinstance(cache, list), "cache must be a list of hidden states for RNN layers."
        new_cache = []
        i = 0
        for (layer_id, layer) in enumerate(self.model.children()):
            if isinstance(layer, nn.RNNBase):
                hidden_state = cache[i] if cache else None
                # make sure hidden state matches the batch size and it is on the correct device
                if hidden_state is not None:
                    if isinstance(hidden_state, tuple) and hidden_state[0].dim() == 3:  # LSTM
                        h, c = hidden_state
                        assert h.size(1) == x.size(0), "Hidden state batch size does not match input batch size. You may need to clear or reset the cache."
                        assert c.size(1) == x.size(0), "Cell state batch size does not match input batch size. You may need to clear or reset the cache."
                        hidden_state = (h.to(x.device), c.to(x.device))
                    elif isinstance(hidden_state, torch.Tensor) and hidden_state.dim() == 3:  # GRU or RNN
                        hidden_state = hidden_state.to(x.device)
                        assert hidden_state.size(1) == x.size(0), "Hidden state batch size does not match input batch size. You may need to clear or reset the cache."
                    else:
                        raise ValueError("Unknown hidden state format in cache.")
                # if it is None, properly initialize it depending on the RNN type
                if hidden_state is None:
                    if isinstance(layer, nn.LSTM):
                        h_size = layer.proj_size if layer.proj_size > 0 else layer.hidden_size
                        h_0 = torch.zeros(layer.num_layers, x.size(0), h_size, device=x.device)
                        c_0 = torch.zeros(layer.num_layers, x.size(0), layer.hidden_size, device=x.device)
                        hidden_state = (h_0, c_0)
                    else:  # GRU or RNN
                        h_0 = torch.zeros(layer.num_layers, x.size(0), layer.hidden_size, device=x.device)
                        hidden_state = h_0
                x, hidden_state = layer(x, hidden_state)
                if isinstance(new_cache, list):
                    new_cache.append(hidden_state)
                i += 1
            else:
                # reshape x to combine batch and sequence dims to apply non-RNN modules
                B, L = x.shape[0], x.shape[1]
                x = x.reshape(B * L, *x.shape[2:])
                x = layer(x)
                # reshape back
                x = x.reshape(B, L, *x.shape[1:])
        return x, new_cache


def clear_cache(cache: Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]], batch_ids: Optional[List[int]] = None) -> Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]:
    """Clear the cached hidden states for the given batch indices. If batch_ids is None, clear all hidden states by simply returning None."""
    if cache is None or batch_ids is None:
        return None
    for hidden_state in cache:
        if hidden_state is None:
            continue
        if isinstance(hidden_state, tuple) and hidden_state[0].dim() == 3:  # LSTM
            h, c = hidden_state
            h[:, batch_ids, :] = 0     # zero out the hidden states for the given batch indices
            c[:, batch_ids, :] = 0     # zero out the cell states for the given batch indices
        elif isinstance(hidden_state, torch.Tensor) and hidden_state.dim() == 3:  # GRU or RNN
            hidden_state[:, batch_ids, :] = 0  # zero out the hidden states for the given batch indices
        else:
            raise ValueError("Unknown hidden state format in cache.")
    return cache


def cache_slice_for_batch_ids(cache: Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]], batch_ids: List[int]) -> Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]:
    """Extract a slice of the cache for the given batch indices."""
    if cache is None:
        return None
    new_cache = []
    for hidden_state in cache:
        if hidden_state is None:
            new_cache.append(None)
            continue
        if isinstance(hidden_state, tuple) and hidden_state[0].dim() == 3:  # LSTM
            h, c = hidden_state
            h_slice = h[:, batch_ids, :]
            c_slice = c[:, batch_ids, :]
            new_cache.append((h_slice, c_slice))
        elif isinstance(hidden_state, torch.Tensor) and hidden_state.dim() == 3:  # GRU or RNN
            h_slice = hidden_state[:, batch_ids, :]
            new_cache.append(h_slice)
        else:
            raise ValueError("Unknown hidden state format in cache.")
    return new_cache


def cache_detach(cache: Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]) -> Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]:
    """Detach the cached hidden states from the computation graph."""
    if cache is None:
        return None
    new_cache = []
    for hidden_state in cache:
        if hidden_state is None:
            new_cache.append(None)
            continue
        if isinstance(hidden_state, tuple):  # LSTM
            h, c = hidden_state
            new_cache.append((h.detach(), c.detach()))
        else:  # GRU or RNN
            new_cache.append(hidden_state.detach())
    return new_cache


def cache_move_to_device(cache: Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]], device: Union[torch.device, str]) -> Optional[List[Union[Tensor, Tuple[Tensor, Tensor]]]]:
    """Move the cached hidden states to the specified device."""
    if cache is None:
        return None
    new_cache = []
    for hidden_state in cache:
        if hidden_state is None:
            new_cache.append(None)
            continue
        if isinstance(hidden_state, tuple):  # LSTM
            h, c = hidden_state
            new_cache.append((h.to(device), c.to(device)))
        else:  # GRU or RNN
            new_cache.append(hidden_state.to(device))
    return new_cache


class ActorBase(nn.Module, abc.ABC):
    """
    Base class for the policy (actor) network in PPO.

    This class wraps a neural network model for policy representation and provides
    utilities for action sampling, log-probability computation, and entropy estimation.
    It supports optional observation normalization using running statistics and can
    operate in deterministic mode for evaluation.

    Attributes:
        model: The underlying neural network (nn.Module) for the policy.
        observation_space: The Gymnasium observation space (Box).
        action_space: The Gymnasium action space (Box or Discrete).
        deterministic: If True, actions are deterministic (mean/argmax).
        norm_obs: If True, normalize observations using running mean/var. ! Critical note: In POMDPs, this may leak information across timesteps.
        obs_mean: Running mean for observation normalization (non-trainable).
        obs_var: Running variance for observation normalization (non-trainable).
        obs_count: Running count for observation normalization (non-trainable).

    Note: Subclasses must implement `sample_action`, `get_entropy`, and `get_logprob`.
    """

    def __init__(self, model, observation_space: Box, action_space: Union[Box, Discrete], deterministic: bool = False, device='cpu', norm_obs: bool = False):
        super(ActorBase, self).__init__()
        self.model = model.to(device)
        self.observation_space = observation_space
        self.action_space = action_space
        self.deterministic = deterministic
        self.norm_obs = norm_obs
        self.obs_mean = nn.Parameter(torch.zeros(observation_space.shape, dtype=torch.float32, device=device), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(observation_space.shape, dtype=torch.float32, device=device), requires_grad=False)
        self.obs_count = nn.Parameter(torch.tensor(0., dtype=torch.float32, device=device), requires_grad=False)
        self.cache = None  # for RNN hidden states if needed

    def update_obs_stats(self, x: Tensor) -> None:
        """Update running normalization statistics for observations using the input batch."""
        if self.norm_obs:
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            _ = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, True)

    def forward_model(self, x: Tensor) -> Tensor:
        """Forward pass through the policy network, applying optional observation normalization."""
        # if dtype is UInt8, convert to float32 and scale to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if isinstance(self.model, SequenceModel):
            # assert shape is 3D
            assert x.dim() >= 3, "Input tensor must be 3D: (batch_size, seq_len, feature_dims...) for SequenceModel."
            # for normalization, flatten batch and seq dims
            B, L = x.shape[0], x.shape[1]
            x = x.reshape(B * L, *x.shape[2:])
            if self.norm_obs:
                x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
            # reshape back
            x = x.reshape(B, L, *x.shape[1:])
            x, self.cache = self.model(x, self.cache)
        else:
            if self.norm_obs:
                x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
            x = self.model(x)
        return x

    def get_subfinal_layer_output(self, x: Tensor) -> Tensor:
        """Get the output from the second last layer of the model (useful for auxiliary tasks)."""
        if isinstance(self.model, SequenceModel):
            raise NotImplementedError("get_subfinal_layer_output is not implemented for SequenceModel.")
        if not isinstance(self.model, nn.Sequential):
            raise NotImplementedError("get_subfinal_layer_output is only implemented for nn.Sequential models.")
        if len(list(self.model.children())) < 2:
            raise ValueError("Model must have at least two layers to get subfinal layer output.")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if self.norm_obs:
            x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
        layers = list(self.model.children())
        for layer in layers[:-1]:
            x = layer(x)
        return x

    @abc.abstractmethod
    def sample_action(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample actions from the policy given observations."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_entropy(self, x) -> Tensor:
        """Compute the entropy of the policy distribution for given observations."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_logprob(self, x: Tensor, action: Tensor) -> Tensor:
        """Compute the log-probability of given actions under the policy for observations."""
        raise NotImplementedError

    def get_kl_div(self, obs, actions, old_logprobs):
        """Approximate KL divergence between old and new policies using log-ratio."""
        log_probs = self.get_logprob(obs, actions)
        log_ratio = log_probs - old_logprobs
        kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio)
        # kl = torch.mean(old_logprobs - log_probs)
        return kl

    def get_policy_loss_and_entropy(self, obs, actions, advantages, old_logprobs, clip_ratio):
        """Compute the clipped PPO policy loss and mean entropy."""
        log_probs = self.get_logprob(obs, actions)
        ratio = torch.exp(log_probs - old_logprobs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        entropy = self.get_entropy(obs).mean()
        return policy_loss, entropy

    def forward(self, x) -> Any:
        """Convenience forward pass: map observations to actions (handles numpy/torch, batching)."""
        # if input is numpy array, convert to torch tensor
        input_is_numpy = isinstance(x, np.ndarray)
        added_time_dim = False
        if input_is_numpy:
            if (x.dtype == np.uint8):
                x = x.astype(np.float32) / 255.0
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.model.parameters()).device)
            if isinstance(self.model, SequenceModel) and len(x.shape) == len(self.observation_space.shape) + 1:
                x = x.unsqueeze(1)  # Add seq_len=1 for batched single-step
                added_time_dim = True
        # if input is not batched, add a batch dimension
        input_is_unbatched = len(x.shape) == len(self.observation_space.shape)
        if input_is_unbatched:
            x = x.unsqueeze(0)
            if isinstance(self.model, SequenceModel):
                x = x.unsqueeze(1)  # add seq_len dimension if RNN
                added_time_dim = True

        # get action
        action, _ = self.sample_action(x)

        if added_time_dim:
            action = action.squeeze(1)  # remove seq_len dim if RNN
        # if input was unbatched, remove the batch dimension
        if input_is_unbatched:
            action = action.squeeze(0)
            # if discrete action space, convert to int
            if isinstance(self.action_space, Discrete):
                action = action.item()
        # if input was numpy, convert output to numpy
        if input_is_numpy and isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

    def clear_cache(self, batch_ids: Optional[List[int]] = None) -> None:
        """Clear the cached hidden states in the underlying model if it is a SequenceModel. If batch_ids is provided, only clear the hidden states for those batch indices."""
        if isinstance(self.model, SequenceModel):
            self.cache = clear_cache(self.cache, batch_ids)

    def evaluate_policy(self, env: Env, num_episodes=100, deterministic=True, seed=0) -> Tuple[List[float], List[float], List[int], List[bool]]:
        """Evaluate the policy by rolling out episodes in a single environment. Default seed is 0. Returns lists of episode rewards, episode costs, episode lengths, and episode successes."""
        self.clear_cache()
        self.eval()
        deterministic_before = self.deterministic
        self.deterministic = deterministic
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        episode_successes = []
        for episode in range(num_episodes):
            print(f"Evaluating episode {episode + 1}/{num_episodes}", end='\r')
            obs, _ = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            total_cost = 0.0
            length = 0
            while not done:
                action = self(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward  # type: ignore
                cost = info.get("cost", 0.0)
                total_cost += cost  # type: ignore
                length += 1
                if done:
                    episode_successes.append(info.get("is_success", False))
                    # no need to clear cache here; the RNN will automatically learn to reset based on an assumed flag in state-space that marks beginning of episodes.
            episode_rewards.append(total_reward)
            episode_costs.append(total_cost)
            episode_lengths.append(length)
        print()
        self.deterministic = deterministic_before  # undo

        return episode_rewards, episode_costs, episode_lengths, episode_successes

    def evaluate_policy_parallel(self, envs: VectorEnv, num_episodes=100, deterministic=True, base_seed=0) -> Tuple[List[float], List[float], List[int], List[bool]]:
        """Evaluate the policy by rolling out episodes in a vectorized environment. Default base_seed is 0, the envs will be seeded: base_seed, base_seed+1, base_seed+2, ... etc. Returns lists of episode rewards, costs, lengths, and success flags."""
        self.clear_cache()
        self.eval()
        deterministic_before = self.deterministic
        self.deterministic = deterministic
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        episode_successes = []
        episode_rew_vec = np.zeros(envs.num_envs)
        episode_cost_vec = np.zeros(envs.num_envs)
        episode_len_vec = np.zeros(envs.num_envs, dtype=int)
        num_envs = envs.num_envs
        obs = envs.reset(seed=base_seed)[0]     # Automatically seeds each env with base_seed + env_index
        while len(episode_rewards) < num_episodes:
            action = self(obs)
            action = action if isinstance(envs.single_action_space, Box) else action.squeeze(-1)
            obs, rewards, terminateds, truncateds, infos = envs.step(action)
            episode_rew_vec += rewards
            episode_cost_vec += infos.get('cost', np.zeros(envs.num_envs))
            episode_len_vec += 1
            for i in range(num_envs):
                if terminateds[i] or truncateds[i]:
                    episode_rewards.append(episode_rew_vec[i])
                    episode_costs.append(episode_cost_vec[i])
                    episode_lengths.append(episode_len_vec[i])
                    # extract is_success from final_info if available
                    if 'final_info' in infos and 'is_success' in infos['final_info']:
                        episode_successes.append(infos['final_info']['is_success'][i])  # type: ignore
                    elif 'final_info' in infos and isinstance(infos['final_info'], list) and 'is_success' in infos['final_info'][i]:    # old gymnasium version
                        episode_successes.append(infos['final_info'][i]['is_success'])  # type: ignore
                    else:
                        episode_successes.append(False)
                    # extract last step cost from final_info if available
                    if 'final_info' in infos and 'cost' in infos['final_info']:
                        episode_cost_vec[i] += infos['final_info']['cost'][i]  # type: ignore
                    elif 'final_info' in infos and isinstance(infos['final_info'], list) and 'cost' in infos['final_info'][i]:
                        episode_cost_vec[i] += infos['final_info'][i]['cost']  # type: ignore
                    episode_rew_vec[i] = 0.0
                    episode_cost_vec[i] = 0.0
                    # no need to clear cache here; the RNN will automatically learn to reset based on an assumed flag in state-space that marks beginning of episodes.
            print(f"Evaluating episodes {len(episode_rewards)}/{num_episodes}", end='\r')
            if len(episode_rewards) >= num_episodes:
                break
        print()
        self.deterministic = deterministic_before  # undo

        return episode_rewards, episode_costs, episode_lengths, episode_successes


class ActorContinuous(ActorBase):
    """
    Continuous action policy using a squashed Gaussian distribution.

    Outputs a mean from the network; samples from N(mean, exp(logstd)), applies tanh
    squashing to [-1, 1], and scales/shifts to match the action space bounds.
    Log-probabilities are adjusted for the tanh transformation and scaling.
    """

    def __init__(self, model, observation_space: Box, action_space: Box, deterministic: bool = False, device='cpu', norm_obs: bool = False, state_dependent_std: bool = False):
        super(ActorContinuous, self).__init__(model, observation_space, action_space, deterministic, device, norm_obs)
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high))
        self.state_dependent_std = state_dependent_std
        self.logstd = nn.Parameter(torch.zeros(1, action_space.shape[0], dtype=torch.float32, device=device), requires_grad=True) if not state_dependent_std else None
        # shift/scale to affine map from [-1, 1] to [low, high] => a = tanh(raw) * scale + shift:
        self.shift = nn.Parameter(torch.tensor(action_space.low + action_space.high, dtype=torch.float32, device=device) / 2.0, requires_grad=False)
        self.scale = nn.Parameter(torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device), requires_grad=False)

    def sample_action(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample continuous actions: reparameterize, tanh-squash, and scale to bounds."""
        mean, logstd = self._get_mean_logstd(x)
        std = torch.exp(logstd)
        # Reparameterized sample with noise, then tanh-squash and scale to bounds.
        noise = torch.randn_like(mean)
        action = mean + std * noise
        log_prob = log_normal_prob(action, mean, std)
        action = torch.tanh(action)
        # Subtract log|det(Jacobian)| for tanh so log_prob matches final action space.
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        action = action * self.scale + self.shift
        info = {'logstd': logstd, 'log_prob': log_prob}
        return action, info

    def get_entropy(self, x):
        """Compute entropy of the squashed Gaussian policy."""
        if self.state_dependent_std:
            # sampling based estimate of entropy
            _, info = self.sample_action(x)
            log_prob = info['log_prob']
            # print("Logprobs for entropy estimation:", log_prob)
            entropy = -log_prob
            return entropy
        else:
            logstd = self._get_logstd(x)
            return gaussian_entropy(logstd)  # technically not exact for squashed Gaussian, but close enough

    def get_logprob(self, x: Tensor, action: Tensor) -> Tensor:
        """Compute log-prob of actions under the squashed Gaussian policy."""
        mean, logstd = self._get_mean_logstd(x)
        std = torch.exp(logstd)
        # Rescale action to [-1, 1]
        action_unshifted_unscaled = (action - self.shift) / (self.scale + 1e-6)
        action_unshifted_unscaled = torch.clamp(action_unshifted_unscaled, -0.999, 0.999)  # otherwise atanh(1) is inf
        action_untanhed = torch.atanh(action_unshifted_unscaled)
        log_prob = log_normal_prob(action_untanhed, mean, std)
        log_prob -= torch.sum(torch.log(1 - action_unshifted_unscaled.pow(2) + 1e-6), dim=-1, keepdim=True)
        return log_prob

    def _get_mean_logstd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the policy mean and logstd from the network."""
        mean = self.forward_model(x)
        if self.state_dependent_std:
            assert mean.shape[-1] % 2 == 0, "For state-dependent std, model output dim must be 2x action dim."
            action_dim = mean.shape[-1] // 2
            logstd = mean[..., action_dim:]
            mean = mean[..., :action_dim]
        else:
            logstd: Tensor = self.logstd  # type: ignore
        logstd = self._logstd_post_process(logstd)
        return mean, logstd

    def _get_logstd(self, x: Tensor) -> Tensor:
        """Get the logstd (from the network if state-dependent)."""
        if self.state_dependent_std:
            mean = self.forward_model(x)
            assert mean.shape[-1] % 2 == 0, "For state-dependent std, model output dim must be 2x action dim."
            action_dim = mean.shape[-1] // 2
            logstd = mean[..., action_dim:]
        else:
            logstd: Tensor = self.logstd  # type: ignore
        logstd = self._logstd_post_process(logstd)
        return logstd

    def _logstd_post_process(self, logstd: Tensor) -> Tensor:
        """Clamp logstd to avoid numerical issues."""
        if self.deterministic:
            logstd = logstd - torch.inf
        logstd = torch.clamp(logstd, -20, 2)    # avoid extreme std values
        return logstd


class ActorDiscrete(ActorBase):
    """
    Discrete action policy using a categorical distribution.

    Outputs logits from the network, applies softmax for probabilities.
    In deterministic mode, logits are inflated to approximate argmax.
    """

    def __init__(self, model, observation_space: Box, action_space: Discrete, deterministic: bool = False, device='cpu', norm_obs: bool = False):
        super(ActorDiscrete, self).__init__(model, observation_space, action_space, deterministic, device, norm_obs)
        assert action_space.n >= 2, "Action space must have at least 2 discrete actions."

    def sample_action(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample discrete actions from the categorical distribution."""
        probs_all, _ = self._get_probs_logprobs_all(x)
        # sample now
        action_dist = torch.distributions.Categorical(probs=probs_all)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        action = action.unsqueeze(-1)  # make it (batch_size, 1) or (batch_size, seq_len, 1)
        log_prob = log_prob.unsqueeze(-1)  # make it (batch_size, 1) or (batch_size, seq_len, 1)
        return action, {'log_prob': log_prob}

    def get_entropy(self, x):
        """Compute entropy of the categorical policy."""
        probs_all, logprobs_all = self._get_probs_logprobs_all(x)
        entropy = -torch.sum(probs_all * logprobs_all, dim=-1)
        return entropy

    def get_logprob(self, x: Tensor, action: Tensor) -> Tensor:
        """Compute log-prob of discrete actions under the categorical policy."""
        _, logprobs_all = self._get_probs_logprobs_all(x)
        return logprobs_all.gather(-1, action)

    def _get_probs_logprobs_all(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get softmax probabilities and log-probs over all actions."""
        logits = self.forward_model(x)
        logits = logits - logits.max(dim=-1, keepdim=True).values  # Stabilize logits by subtracting max (per batch row) before softmax/log_softmax.
        if self.deterministic:
            logits = logits * 1e6  # make it very large to approximate argmax
        probs_all = F.softmax(logits, dim=-1)   # (batch_size, action_dim) or (batch_size, seq_len, action_dim)
        logprobs_all = F.log_softmax(logits, dim=-1)  # (batch_size, action_dim) or (batch_size, seq_len, action_dim)
        return probs_all, logprobs_all


def Actor(model, observation_space: Box, action_space: Union[Box, Discrete], deterministic: bool = False, device='cpu', norm_obs: bool = False, state_dependent_std: bool = False) -> ActorBase:
    """Factory function to create an appropriate Actor subclass based on the action space."""
    # raise a warning if norm_obs is True and the model is a SequenceModel
    if norm_obs and isinstance(model, SequenceModel):
        print("Warning: Observation normalization is enabled for a SequenceModel. In POMDPs, this may leak information across timesteps.", file=sys.stderr)
    if isinstance(action_space, Box):
        return ActorContinuous(model, observation_space, action_space, deterministic, device, norm_obs, state_dependent_std=state_dependent_std)
    elif isinstance(action_space, Discrete):
        return ActorDiscrete(model, observation_space, action_space, deterministic, device, norm_obs)
    else:
        raise NotImplementedError("Only Box and Discrete action spaces are supported.")


class Critic(nn.Module):
    """
    Value function (critic) network for state-value estimation V(s).

    Estimates the expected discounted return from a given state under the current policy.
    Supports optional observation normalization using running statistics.

    Attributes:
        model: The underlying neural network (nn.Module) for value prediction.
        norm_obs: If True, normalize observations using running mean/var. ! Critical note: In POMDPs, this may leak information across timesteps.
        obs_mean: Running mean for observation normalization (non-trainable).
        obs_var: Running variance for observation normalization (non-trainable).
        obs_count: Running count for observation normalization (non-trainable).
    """

    def __init__(self, model, observation_space: Box, device='cpu', norm_obs: bool = False):
        super(Critic, self).__init__()
        self.model = model
        self.norm_obs = norm_obs
        self.obs_mean = nn.Parameter(torch.zeros(observation_space.shape, dtype=torch.float32, device=device), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(observation_space.shape, dtype=torch.float32, device=device), requires_grad=False)
        self.obs_count = nn.Parameter(torch.tensor(0., dtype=torch.float32, device=device), requires_grad=False)
        self.cache = None  # for RNN hidden states if needed
        if norm_obs and isinstance(model, SequenceModel):
            print("Warning: Observation normalization is enabled for a SequenceModel. In POMDPs, this may leak information across timesteps.", file=sys.stderr)

    def update_obs_stats(self, x: Tensor) -> None:
        """Update running normalization statistics for observations using the input batch."""
        if self.norm_obs:
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            _ = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, True)

    def get_loss(self, states, old_values, advantages):
        """Compute the MSE loss for value regression on target returns (old_values + advantages)."""
        values = (old_values + advantages).detach()
        values_pred = self.get_value(states)
        return F.mse_loss(values_pred, values)

    def get_value(self, x: Tensor) -> Tensor:
        """Predict state values, applying optional observation normalization."""
        # if dtype is UInt8, convert to float32 and scale to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if isinstance(self.model, SequenceModel):
            # assert shape is 3D
            assert x.dim() >= 3, "Input tensor must be 3D: (batch_size, seq_len, feature_dims...) for SequenceModel."
            # for normalization, flatten batch and seq dims
            B, L = x.shape[0], x.shape[1]
            x = x.reshape(B * L, *x.shape[2:])
            if self.norm_obs:
                x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
            # reshape back
            x = x.reshape(B, L, *x.shape[1:])
            x, self.cache = self.model(x, self.cache)
        else:
            if self.norm_obs:
                x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
            x = self.model(x)
        return x

    def get_subfinal_layer_output(self, x: Tensor) -> Tensor:
        """Get the output of the layer before the final output layer (for use in e.g., cost critics)."""
        if isinstance(self.model, SequenceModel):
            raise NotImplementedError("get_subfinal_layer_output is not implemented for SequenceModel.")
        else:
            # assuming model is nn.Sequential
            if not isinstance(self.model, nn.Sequential):
                raise NotImplementedError("get_subfinal_layer_output is only implemented for nn.Sequential models.")
            if len(self.model) < 2:
                raise ValueError("Model must have at least 2 layers to get subfinal layer output.")
            # if dtype is UInt8, convert to float32 and scale to [0, 1]
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            if self.norm_obs:
                x = normalize_and_update_stats(self.obs_mean, self.obs_var, self.obs_count, x, False)
            for layer in list(self.model.children())[:-1]:
                x = layer(x)
            return x

    def clear_cache(self, batch_ids: Optional[List[int]] = None) -> None:
        """Clear the cached hidden states in the underlying model if it is a SequenceModel. If batch_ids is provided, only clear the hidden states for those batch indices."""
        if isinstance(self.model, SequenceModel):
            self.cache = clear_cache(self.cache, batch_ids)


class PPO:
    """
    Proximal Policy Optimization (PPO) trainer.

    This class implements the PPO algorithm, including trajectory collection, advantage
    estimation via GAE, and clipped policy/value updates. It supports vectorized
    environments in SAME_STEP autoreset mode for efficient data collection.

    Key Features:
        - Supports continuous and discrete actions via Actor subclasses.
        - Optional Constrained MDP (CMDP) mode with Lagrange multiplier for cost constraints.
        - Separate devices for inference (rollouts) and training (updates) for performance.
        - Comprehensive logging of metrics (returns, losses, KL, etc.) for analysis.
        - Optional annealing of hyperparameters (LR, entropy coef, clip ratio).
        - Normalization of observations, rewards, and advantages for stability.

    Usage:
        1. Create a vectorized environment e.g., AsyncVectorEnv with SAME_STEP autoreset (skip AutoresetMode if using older gymnasium versions).
        2. Define actor and critic networks (nn.Module).
        3. Instantiate PPO with envs, actor, critic, and hyperparameters.
        4. Call `train(num_iterations)` to run training loops.
        5. Access `stats` dict for logged metrics (e.g., mean_returns, losses).

    CMDP Mode:
        - Requires a cost_critic and cost_threshold > 0.
        - Uses Lagrange multiplier to penalize cost advantages in policy updates.
        - Supports constraining discounted or undiscounted costs via `constrain_undiscounted_cost`.

    Example:
        envs = AsyncVectorEnv([...], autoreset_mode=AutoresetMode.SAME_STEP)    # skip AutoresetMode if using older gymnasium versions
        actor = Actor(mlp_model, obs_space, act_space, norm_obs=True)
        critic = Critic(mlp_model, obs_space, norm_obs=True)
        ppo = PPO(envs, actor, critic, iters=1000, gamma=0.99, clip_ratio=0.2)
        ppo.train(1000)
        plt.plot(ppo.stats['mean_returns'])
    """
    def __init__(self,
                 envs: VectorEnv,
                 actor: ActorBase,
                 critic: Critic,
                 iters: int,
                 cmdp_mode: bool = False,
                 cost_critic: Optional[Critic] = None,  # required if cmdp_mode is True
                 cost_threshold: float = 0.0,           # must be > 0 if cmdp_mode is True
                 constrain_undiscounted_cost: bool = False,  # if True, use undiscounted cost to update lagrange multiplier, else use discounted cost
                 lagrange_lr: float = 0.01,
                 lagrange_max: float = 1000.0,
                 moving_cost_estimate_step_size: float = 0.01,  # step size for moving average estimation of cost per episode. Formula: new_estimate = (1-step_size)*old_estimate + step_size*new_observation. Whether the estimate is discounted or undiscounted depends on constrain_undiscounted_cost.
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 decay_lr: bool = False,
                 min_lr: float = 1.25e-5,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 nsteps: int = 2048,
                 nepochs: int = 10,
                 batch_size: int = 64,
                 context_length: Optional[int] = None,  # if using RNNs, the context length for BPTT. It must be less than or equal to nsteps as well as less than or equal to batch_size. If None, set to minimum of nsteps and batch_size.
                 clip_ratio: float = 0.2,
                 decay_clip_ratio: bool = False,
                 min_clip_ratio: float = 0.01,
                 target_kl: float = np.inf,
                 early_stop_critic: bool = False,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.0,
                 decay_entropy_coef: bool = False,
                 normalize_advantages: bool = True,
                 clipnorm: float = 0.5,
                 norm_rewards: bool = False,
                 adam_weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-7,
                 base_seed: int = 42,  # base seed for envs, the envs will be seeded: base_seed, base_seed+1, base_seed+2, ... etc.
                 inference_device: str = 'cpu',
                 training_device: str = 'cpu',
                 verbose: bool = True,
                 custom_info_keys_to_log_at_episode_end: List[str] = [],):  # keys in info dict to log at episode end
        """
        Initialize the PPO trainer with environments, networks, and hyperparameters.

        Args:
            envs: Vectorized environment (must use SAME_STEP autoreset mode).
            actor: Policy network (ActorBase instance).
            critic: Value network (Critic instance).
            iters: Total number of training iterations.
            cmdp_mode: If True, enable Constrained MDP mode with cost constraints.
            cost_critic: Cost value network (required if cmdp_mode=True).
            cost_threshold: Target average cost per episode (required if cmdp_mode=True).
            constrain_undiscounted_cost: If True, constrain undiscounted costs for Lagrange update.
            lagrange_lr: Learning rate for Lagrange multiplier update.
            lagrange_max: Maximum value for Lagrange multiplier.
            moving_cost_estimate_step_size: Step size for moving average estimation of cost per episode. Formula: new_estimate = (1-step_size) x old_estimate + step_size x new_observation. Whether the estimate is discounted or undiscounted depends on constrain_undiscounted_cost.
            actor_lr: Initial learning rate for actor optimizer.
            critic_lr: Initial learning rate for critic optimizer.
            decay_lr: If True, anneal learning rates linearly over iterations.
            min_lr: Minimum learning rate after annealing.
            gamma: Discount factor for returns and advantages.
            lam: GAE lambda for advantage estimation.
            nsteps: Steps per trajectory rollout (per env).
            nepochs: Number of SGD epochs per iteration.
            batch_size: Minibatch size for SGD.
            context_length: For RNNs, the context length for BPTT. Must be <= nsteps and <= batch_size. If None, set to min(nsteps, batch_size).
            clip_ratio: PPO clipping parameter (epsilon).
            decay_clip_ratio: If True, anneal clip ratio linearly.
            min_clip_ratio: Minimum clip ratio after annealing.
            target_kl: Target KL divergence for early stopping.
            early_stop_critic: If True, stop critic updates when actor early-stops.
            value_loss_coef: Coefficient for value loss in total objective.
            entropy_coef: Initial coefficient for entropy regularization.
            decay_entropy_coef: If True, anneal entropy coef linearly.
            normalize_advantages: If True, normalize advantages before policy update.
            clipnorm: Gradient norm clipping value (0.5 by default).
            norm_rewards: If True, normalize rewards by running std of discounted returns.
            adam_weight_decay: Weight decay for Adam optimizers.
            adam_epsilon: Epsilon for Adam optimizers.
            inference_device: Device for rollout/inference (e.g., 'cpu').
            base_seed: Base seed for envs; envs will be seeded sequentially at the first reset call e.g., base_seed, base_seed+1, ...
            training_device: Device for parameter updates (e.g., 'cuda').
            custom_info_keys_to_log_at_episode_end: List of env info keys to log per episode. Their 100-window averages will be tracked in stats and printed during training.

        Raises:
            AssertionError: If envs not in SAME_STEP mode (in recent gymnasium versions, older versions are usually already similar to SAME_STEP), or CMDP params invalid.
        """
        try:
            from gymnasium.vector import AutoresetMode
            assert envs.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP, "VectorEnv must be in SAME_STEP autoreset mode"
        except (KeyError, ImportError):
            print("Warning: Could not verify that VectorEnv is in SAME_STEP autoreset mode. Ensure that envs are created with autoreset_mode=AutoresetMode.SAME_STEP in recent gymnasium versions. The reset logic in older versions is usually already similar to SAME_STEP.", file=sys.stderr)

        self.envs = envs
        self.actor = actor.to(training_device)
        self.critic = critic.to(training_device)
        self.cmdp_mode = cmdp_mode
        if self.cmdp_mode:
            assert cost_critic is not None, "cost_critic must be provided in CMDP mode"
            assert cost_threshold > 0.0, "cost_threshold must be > 0 in CMDP mode"
            if not normalize_advantages:
                print("Warning: It is recommended to normalize advantages when using CMDP mode so that the scale of reward and cost advantages are similar before combining them.", file=sys.stderr)
            self.cost_critic = cost_critic.to(training_device)
        self.iters = iters
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=adam_weight_decay, eps=adam_epsilon)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=adam_weight_decay, eps=adam_epsilon)
        if self.cmdp_mode:
            self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=critic_lr, weight_decay=adam_weight_decay, eps=adam_epsilon)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.decay_lr = decay_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.nepochs = nepochs
        self.batch_size = batch_size
        assert (nsteps * envs.num_envs) % batch_size == 0, "nsteps * num_envs must be divisible by batch_size"
        self.context_length = context_length if context_length is not None else min(nsteps, batch_size)
        self.is_rucur = isinstance(self.actor.model, SequenceModel) or isinstance(self.critic.model, SequenceModel) or (self.cmdp_mode and isinstance(self.cost_critic.model, SequenceModel))
        if self.is_rucur:
            # ensure all models are SequenceModel
            assert isinstance(self.actor.model, SequenceModel) and isinstance(self.critic.model, SequenceModel), "All actor and critic models must be SequenceModel when using RNNs"
            if self.cmdp_mode:
                assert isinstance(self.cost_critic.model, SequenceModel), "All actor and critic models must be SequenceModel when using RNNs"
            # assert context_length is valid
            assert self.context_length <= self.nsteps, "context_length must be less than or equal to nsteps"
            # ensure context_length divides nsteps for simplicity
            assert self.nsteps % self.context_length == 0, "When using RNNs, nsteps must be divisible by context_length for simplicity"
            # ensure batch_size divides context_length times num_envs
            assert (self.context_length * envs.num_envs) % self.batch_size == 0, "When using RNNs, context_length * num_envs must be divisible by batch_size for simplicity"
        self.clip_ratio = clip_ratio
        self.decay_clip_ratio = decay_clip_ratio
        self.min_clip_ratio = min_clip_ratio
        self.target_kl = target_kl
        self.early_stop_critic = early_stop_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.decay_entropy_coef = decay_entropy_coef
        self.normalize_advantages = normalize_advantages
        self.clipnorm = clipnorm
        self.norm_rewards = norm_rewards
        self.base_seed = base_seed
        self.inference_device = inference_device
        self.training_device = training_device
        self.custom_info_keys_to_log_at_episode_end = custom_info_keys_to_log_at_episode_end
        self.cost_threshold = cost_threshold
        self.constrain_undiscounted_cost = constrain_undiscounted_cost
        self.lagrange_lr = lagrange_lr
        self.lagrange_max = lagrange_max
        self.moving_cost_estimate_step_size = moving_cost_estimate_step_size
        self.lagrange = 0.0  # Lagrange multiplier
        self.moving_average_cost = 0.0  # moving average estimate of cost per episode, discounted or undiscounted depending on constrain_undiscounted_cost
        self.verbose = verbose

        # stats
        self.stats = {
            'iterations': 0,
            'total_timesteps': 0,
            'total_episodes': 0,
            'timestepss': [],           # after each iteration
            'episodess': [],            # after each iteration
            'losses': [],               # after each iteration
            'actor_losses': [],         # after each iteration
            'critic_losses': [],        # after each iteration
            'cost_losses': [],          # after each iteration
            'entropies': [],            # after each iteration
            'kl_divs': [],              # after each iteration
            'values': [],               # mean value of states in the batch after each iteration
            'cost_values': [],          # mean cost value of states in the batch after each iteration
            'logprobs': [],             # mean logprob of actions in the batch after each iteration (its negative is the true mean entropy of the actions in the batch)
            'logstds': [],              # mean logstd of actions in the batch after each iteration
            'actor_lrs': [],            # adjusted per iteration
            'critic_lrs': [],           # adjusted per iteration
            'entropy_coefs': [],        # adjusted per iteration
            'clip_ratios': [],          # adjusted per iteration
            'lagranges': [],            # after each iteration
            'returns': [],              # return of each trajectory collected so far
            'discounted_returns': [],   # discounted return of each trajectory collected so far
            'discounted_costs': [],     # discounted cost of each trajectory collected so far
            'lengths': [],              # length of each trajectory collected so far
            'total_costs': [],          # total cost of each trajectory collected so far (if env provides this info)
            'successes': [],            # success of each trajectory collected so far (if env provides this info)
            'mean_returns': [],         # mean return of latest 100 trajectories after each iteration
            'mean_discounted_returns': [],  # mean discounted return of latest 100 trajectories after each iteration
            'mean_discounted_costs': [],    # mean discounted cost of latest 100 trajectories after each iteration
            'mean_lengths': [],         # mean length of latest 100 trajectories after each iteration
            'mean_total_costs': [],     # mean total cost of latest 100 trajectories after each iteration (if env provides this info)
            'mean_successes': [],       # mean success of latest 100 trajectories after each iteration (if env provides this info)
        }
        self.stats['info_keys'] = {key: [] for key in (self.custom_info_keys_to_log_at_episode_end)}
        self.stats['info_key_means'] = {key: [] for key in (self.custom_info_keys_to_log_at_episode_end)}  # mean of latest 100 episodes after each iteration

        state_shape: Tuple[int, ...] = envs.single_observation_space.shape  # type: ignore
        if envs.single_observation_space.dtype == np.uint8:
            obs_dtype = torch.uint8
        else:
            obs_dtype = torch.float32
        if isinstance(envs.single_action_space, Discrete):
            action_dim = 1
            action_dtype = torch.int64
        elif isinstance(envs.single_action_space, Box):
            action_dim = envs.single_action_space.shape[0]  # type: ignore
            action_dtype = torch.float32
        else:
            raise NotImplementedError("Only Box and Discrete action spaces are supported.")
        M, N = self.nsteps, self.envs.num_envs
        # Experience buffers (shape: [N_envs, nsteps, ...]) for obs, actions, logprobs,
        # rewards/costs, done flags, and advantages/values. Flattened later for SGD.
        self.obs_buf = torch.zeros((N, M, *state_shape), dtype=obs_dtype, device=self.training_device)
        self.actions_buf = torch.zeros((N, M, action_dim), dtype=action_dtype, device=self.training_device)
        self.log_probs_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.rewards_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.costs_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.terminateds_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.truncateds_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.values_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.cost_values_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.advantages_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.cost_advantages_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)

        # data buffers for collecting trajectories, if using a different device for inference
        if self.inference_device != self.training_device:
            self.obs_buf_inference = self.obs_buf.clone().to(self.inference_device)
            self.actions_buf_inference = self.actions_buf.clone().to(self.inference_device)
            self.log_probs_buf_inference = self.log_probs_buf.clone().to(self.inference_device)
            self.rewards_buf_inference = self.rewards_buf.clone().to(self.inference_device)
            self.costs_buf_inference = self.costs_buf.clone().to(self.inference_device)
            self.terminateds_buf_inference = self.terminateds_buf.clone().to(self.inference_device)
            self.truncateds_buf_inference = self.truncateds_buf.clone().to(self.inference_device)
        else:
            self.obs_buf_inference = self.obs_buf
            self.actions_buf_inference = self.actions_buf
            self.log_probs_buf_inference = self.log_probs_buf
            self.rewards_buf_inference = self.rewards_buf
            self.costs_buf_inference = self.costs_buf
            self.terminateds_buf_inference = self.terminateds_buf
            self.truncateds_buf_inference = self.truncateds_buf

        self.returns_buffer = np.zeros((N,), dtype=np.float32)
        self.discounted_returns_buffer = np.zeros((N,), dtype=np.float32)
        self.discounted_costs_buffer = np.zeros((N,), dtype=np.float32)
        self.lengths_buffer = np.zeros((N,), dtype=np.int32)
        self.costs_total_buffer = np.zeros((N,), dtype=np.float32)

        # clear caches in actor and critics
        self.actor.clear_cache()
        self.critic.clear_cache()
        if self.cmdp_mode:
            self.cost_critic.clear_cache()

        # context before this iteration for RNNs
        self.actor_inference_context_initial = None
        self.critic_context_initial = None
        self.cost_critic_context_initial = None

        # Optional copy of the actor on the inference device (kept in sync after updates).
        if self.inference_device != self.training_device:
            self.actor_inference = copy.deepcopy(self.actor).to(self.inference_device)
            self.critic_inference = copy.deepcopy(self.critic).to(self.inference_device)
        else:
            self.actor_inference = self.actor
            self.critic_inference = self.critic

        # reset all envs and store initial observation
        self.obs: torch.Tensor = torch.tensor(envs.reset(seed=self.base_seed)[0], dtype=obs_dtype, device=self.inference_device)    # Automatically seeds each env with base_seed + env_index

    def collect_trajectories(self) -> None:
        """Collect rollout trajectories from the vector environment using the inference actor."""
        self.actor_inference.eval()
        self.actor_inference.cache = copy.deepcopy(self.actor_inference_context_initial)  # set context for RNNs
        for t in range(self.nsteps):
            with torch.no_grad():
                # act and step
                obs = self.obs.unsqueeze(1) if self.is_rucur else self.obs  # add seq_len dimension if RNN
                a_t, a_t_info = self.actor_inference.sample_action(obs)
                logp_t = a_t_info['log_prob']
                if self.is_rucur:
                    a_t = a_t.squeeze(1)  # remove seq_len dimension
                    logp_t = logp_t.squeeze(1)  # remove seq_len dimension
                _a_t = a_t if isinstance(self.envs.single_action_space, Box) else a_t.squeeze(-1)
                obs_next_potentially_resetted, r_t, terminated_t, truncated_t, infos = self.envs.step(_a_t.cpu().numpy())

                # store data
                self.obs_buf_inference[:, t, ...] = self.obs
                self.actions_buf_inference[:, t, :] = a_t
                self.log_probs_buf_inference[:, t, :] = logp_t
                self.rewards_buf_inference[:, t, 0] = torch.tensor(r_t, dtype=torch.float32, device=self.inference_device)
                # If CMDP: env must provide "cost" (either per-step or via "final_info").
                # We aggregate both discounted and undiscounted views for different constraints.
                c_t = infos.get('cost', np.zeros(self.envs.num_envs))
                if 'final_info' in infos and 'cost' in infos['final_info']:  # some episode ended and vecenv provided final cost info for that env and zeros for others in a vectorized way (newer gymnasium versions)
                    c_t += infos['final_info']['cost']
                elif 'final_info' in infos and isinstance(infos['final_info'], list):  # vecenv provided final info as a list of dicts (older gymnasium versions)
                    for i in range(self.envs.num_envs):
                        if (truncated_t[i] or terminated_t[i]) and 'cost' in infos['final_info'][i]:
                            c_t[i] += infos['final_info'][i]['cost']
                self.costs_buf_inference[:, t, 0] = torch.tensor(c_t, dtype=torch.float32, device=self.inference_device)
                self.terminateds_buf_inference[:, t, 0] = torch.tensor(terminated_t, dtype=torch.float32, device=self.inference_device)
                self.truncateds_buf_inference[:, t, 0] = torch.tensor(truncated_t, dtype=torch.float32, device=self.inference_device)

                # update current obs
                self.obs.data.copy_(torch.tensor(obs_next_potentially_resetted, device=self.inference_device), non_blocking=True)

                # update returns, lengths, costs
                self.discounted_returns_buffer += r_t * np.power(self.gamma, self.lengths_buffer)
                self.discounted_costs_buffer += c_t * np.power(self.gamma, self.lengths_buffer)
                self.returns_buffer += r_t
                self.lengths_buffer += 1
                self.costs_total_buffer += c_t
                for i in range(self.envs.num_envs):
                    if terminated_t[i] or truncated_t[i]:
                        self.stats['returns'].append(self.returns_buffer[i])
                        self.stats['discounted_returns'].append(self.discounted_returns_buffer[i])
                        self.stats['discounted_costs'].append(self.discounted_costs_buffer[i])
                        self.stats['lengths'].append(self.lengths_buffer[i])
                        self.stats['total_costs'].append(self.costs_total_buffer[i])
                        if self.cmdp_mode:
                            # update moving average estimate of cost per episode
                            if self.constrain_undiscounted_cost:
                                if (len(self.stats['total_costs']) == 1):
                                    self.moving_average_cost = self.costs_total_buffer[i]
                                else:
                                    self.moving_average_cost = (1 - self.moving_cost_estimate_step_size) * self.moving_average_cost + self.moving_cost_estimate_step_size * self.costs_total_buffer[i]
                            else:
                                if (len(self.stats['discounted_costs']) == 1):
                                    self.moving_average_cost = self.discounted_costs_buffer[i]
                                else:
                                    self.moving_average_cost = (1 - self.moving_cost_estimate_step_size) * self.moving_average_cost + self.moving_cost_estimate_step_size * self.discounted_costs_buffer[i]
                        if 'final_info' in infos and 'is_success' in infos['final_info']:
                            self.stats['successes'].append(infos['final_info']['is_success'][i])  # type: ignore
                        self.returns_buffer[i] = 0.0
                        self.discounted_returns_buffer[i] = 0.0
                        self.discounted_costs_buffer[i] = 0.0
                        self.lengths_buffer[i] = 0
                        self.costs_total_buffer[i] = 0.0
                        self.stats['total_episodes'] += 1
                        # Don't reset the cache here; the RNN will automatically learn to reset based on an assumed flag in state-space that marks beginning of episodes.

                        for key in self.custom_info_keys_to_log_at_episode_end:
                            key_found = False
                            if 'final_info' in infos and key in infos['final_info']:
                                self.stats['info_keys'][key].append(infos['final_info'][key][i])    # type: ignore
                                key_found = True
                            elif 'final_info' in infos and isinstance(infos['final_info'], list):   # older gymnasium versions
                                if key in infos['final_info'][i]:
                                    self.stats['info_keys'][key].append(infos['final_info'][i][key])
                                    key_found = True
                            if not key_found:
                                # raise warning only first time
                                if len(self.stats['info_keys'][key]) == 0:
                                    print(f"Warning: info key '{key}' not found in env info dictionary. Ignoring it. This warning will not be repeated.", file=sys.stderr)
                                self.stats['info_keys'][key].append(np.nan)

                if self.verbose:
                    print(f"Collected steps {(t + 1) * self.envs.num_envs}/{self.nsteps * self.envs.num_envs}", end='\r')
        if self.verbose:
            terminal_width = shutil.get_terminal_size((80, 20)).columns
            print(' ' * terminal_width, end='\r')  # erase the progress log

        # if using different device for inference, copy data to training device
        if self.inference_device != self.training_device:
            self.obs_buf.data.copy_(self.obs_buf_inference.data, non_blocking=True)
            self.actions_buf.data.copy_(self.actions_buf_inference.data, non_blocking=True)
            self.log_probs_buf.data.copy_(self.log_probs_buf_inference.data, non_blocking=True)
            self.rewards_buf.data.copy_(self.rewards_buf_inference.data, non_blocking=True)
            self.costs_buf.data.copy_(self.costs_buf_inference.data, non_blocking=True)
            self.terminateds_buf.data.copy_(self.terminateds_buf_inference.data, non_blocking=True)
            self.truncateds_buf.data.copy_(self.truncateds_buf_inference.data, non_blocking=True)

        # update stats
        self.stats['total_timesteps'] += self.nsteps * self.envs.num_envs
        self.stats['timestepss'].append(self.stats['total_timesteps'])
        self.stats['episodess'].append(self.stats['total_episodes'])
        # Rolling window stats (last 100 episodes): mean return/length/cost/success.
        # Also track custom info keys from env for handy metrics.
        mean_return = np.mean(self.stats['returns'][-100:]) if len(self.stats['returns']) > 0 else 0.0
        mean_discounted_return = np.mean(self.stats['discounted_returns'][-100:]) if len(self.stats['discounted_returns']) > 0 else 0.0
        mean_discounted_cost = np.mean(self.stats['discounted_costs'][-100:]) if len(self.stats['discounted_costs']) > 0 else 0.0
        mean_length = np.mean(self.stats['lengths'][-100:]) if len(self.stats['lengths']) > 0 else 0.0
        mean_total_cost = np.mean(self.stats['total_costs'][-100:]) if len(self.stats['total_costs']) > 0 else 0.0
        mean_success = np.mean(self.stats['successes'][-100:]) if len(self.stats['successes']) > 0 else 0.0
        self.stats['mean_returns'].append(mean_return)
        self.stats['mean_discounted_returns'].append(mean_discounted_return)
        self.stats['mean_discounted_costs'].append(mean_discounted_cost)
        self.stats['mean_lengths'].append(mean_length)
        self.stats['mean_total_costs'].append(mean_total_cost)
        self.stats['mean_successes'].append(mean_success)
        for key in self.custom_info_keys_to_log_at_episode_end:
            key_mean = np.nanmean(self.stats['info_keys'][key][-100:]) if len(self.stats['info_keys'][key]) > 0 else 0.0
            self.stats['info_key_means'][key].append(key_mean)

    def compute_values_advantages(self) -> None:
        """Compute value estimates and GAE advantages for rewards (and costs in CMDP mode)."""
        self.critic.eval()
        self.critic.cache = copy.deepcopy(self.critic_context_initial)  # set context for RNNs
        self.actor.eval()
        self.actor.cache = copy.deepcopy(self.actor_inference_context_initial)  # set context for RNNs
        if self.cmdp_mode:
            self.cost_critic.eval()
            self.cost_critic.cache = copy.deepcopy(self.cost_critic_context_initial)  # set context for RNNs
        with torch.no_grad():
            obs_buf_cat_cur_obs = torch.cat([self.obs_buf, self.obs.to(self.obs_buf.device).unsqueeze(1)], dim=1)   # (N, M+1, obs_dim)
            if not self.is_rucur:
                obs_buf_cat_cur_obs = obs_buf_cat_cur_obs.reshape(-1, *obs_buf_cat_cur_obs.shape[2:])  # (N*(M+1), obs_dim)

            v_cat_v_cur = self.critic.get_value(obs_buf_cat_cur_obs)    # (N*(M+1), 1) or (N, M+1, 1) depending on whether RNN
            v_cat_v_cur = v_cat_v_cur.reshape(self.envs.num_envs, self.nsteps + 1, 1)
            v = v_cat_v_cur[:, :-1, :]  # (N, M, 1)
            v_next = v_cat_v_cur[:, 1:, :]  # (N, M, 1)

            advantages = self.rewards_buf + self.gamma * (1.0 - self.terminateds_buf) * v_next - v  # (N, M, 1)

            # ! Important: Since we did not handle episode boundaries properly, v_next is not exactly value of obs_prime when dones are 1. This is not an issue for terminated episodes because v_next does not contribute to advantages, but for truncated episodes, v_next is being used for bootstrapping. We should set advantages to zero where episodes are truncated so that losses are not affected. We do lose some data here (insignificant if timelimits are large), but it's better than having wrong advantages. We are not going to use a more complex method to use every single data point since it adds significantly more computation complexity and moreover, makes it really hard to deal with RNNs. For RNNs, we assume that the state space has a flag that marks the beginning of episodes, so that the RNN automatically learns to reset its hidden states at episode boundaries.
            advantages *= (1.0 - self.truncateds_buf)

            adv_next = torch.zeros((self.envs.num_envs, 1), dtype=torch.float32, device=self.training_device)
            dones = (self.terminateds_buf + self.truncateds_buf).clamp(max=1.0)  # (N, M, 1)
            for t in reversed(range(self.nsteps)):
                # Standard backward-time GAE(λ): A_t = δ_t + γλ (1 - done_t) A_{t+1}
                advantages[:, t, :] += (1.0 - dones[:, t, :]) * self.gamma * self.lam * adv_next
                adv_next = advantages[:, t, :]

            self.values_buf.data.copy_(v.data, non_blocking=True)
            self.advantages_buf.data.copy_(advantages.data, non_blocking=True)

            if self.cmdp_mode:
                c_cat_c_cur = self.cost_critic.get_value(obs_buf_cat_cur_obs)    # (N*(M+1), 1) or (N, M+1, 1) depending on whether RNN
                c_cat_c_cur = c_cat_c_cur.reshape(self.envs.num_envs, self.nsteps + 1, 1)
                c = c_cat_c_cur[:, :-1, :]  # (N, M, 1)
                c_next = c_cat_c_cur[:, 1:, :]  # (N, M, 1)
                cost_advantages = self.costs_buf + self.gamma * (1.0 - self.terminateds_buf) * c_next - c  # (N, M, 1)
                cost_advantages *= (1.0 - self.truncateds_buf)  # zero out advantages where episodes are truncated (see comment above)
                cost_adv_next = torch.zeros((self.envs.num_envs, 1), dtype=torch.float32, device=self.training_device)
                for t in reversed(range(self.nsteps)):
                    cost_advantages[:, t, :] += (1.0 - dones[:, t, :]) * self.gamma * self.lam * cost_adv_next
                    cost_adv_next = cost_advantages[:, t, :]

                self.cost_values_buf.data.copy_(c.data, non_blocking=True)
                self.cost_advantages_buf.data.copy_(cost_advantages.data, non_blocking=True)

    def loss(self, obs: Tensor, actions: Tensor, old_values: Tensor, advantages: Tensor, old_logprobs: Tensor, old_cost_values: Tensor, cost_advantages: Tensor, clip_ratio, actor_coeff=1.0, value_coeff=0.5, cost_coeff=0.5, entropy_coeff=0.0) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Compute the combined PPO loss (policy + value + entropy + optional cost)."""
        value_loss = self.critic.get_loss(obs, old_values, advantages)
        cost_loss = self.cost_critic.get_loss(obs, old_cost_values, cost_advantages) if self.cmdp_mode else None
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean().detach()) / (advantages.std().detach() + 1e-3)
            if self.cmdp_mode:
                cost_advantages = (cost_advantages - cost_advantages.mean().detach()) / (cost_advantages.std().detach() + 1e-3)
        if self.cmdp_mode:
            advantages -= self.lagrange * cost_advantages
            if self.normalize_advantages:   # normalize again after combining
                advantages = (advantages - advantages.mean().detach()) / (advantages.std().detach() + 1e-3)  # again
        actor_loss, entropy = self.actor.get_policy_loss_and_entropy(obs, actions, advantages, old_logprobs, clip_ratio)
        if cost_loss is not None:
            total_loss = actor_coeff * actor_loss + value_coeff * value_loss + cost_coeff * cost_loss - entropy_coeff * entropy
        else:
            total_loss = actor_coeff * actor_loss + value_coeff * value_loss - entropy_coeff * entropy
        return total_loss, actor_loss, value_loss, cost_loss, entropy

    def train_one_iteration(self):
        """Perform one full PPO iteration: collect data, optionally normalize rewards/costs, update obs-normalization stats, compute advantages, anneal hyperparameters, update networks, log stats."""
        # collect data
        self.collect_trajectories()
        # update normalization stats
        if self.norm_rewards and len(self.stats['discounted_returns']) > 1:
            reward_std = np.std(self.stats['discounted_returns'])
            if reward_std >= 1e-3:
                self.rewards_buf /= reward_std  # type: ignore
            if self.cmdp_mode:
                costs_std = np.std(self.stats['discounted_costs'])
                if costs_std > 1e-3:
                    self.costs_buf /= costs_std  # type: ignore
        if self.actor.norm_obs:
            self.actor.update_obs_stats(self.obs_buf.reshape(-1, *self.obs_buf.shape[2:]))
        if self.critic.norm_obs:
            self.critic.update_obs_stats(self.obs_buf.reshape(-1, *self.obs_buf.shape[2:]))
        if self.cmdp_mode and self.cost_critic.norm_obs:
            self.cost_critic.update_obs_stats(self.obs_buf.reshape(-1, *self.obs_buf.shape[2:]))

        # compute values and advantages
        self.compute_values_advantages()
        # do annealing updates of lr, entropy coef, clip ratio
        entropy_coef = self.entropy_coef
        if self.decay_entropy_coef:
            entropy_coef = self.entropy_coef * (1.0 - float(self.stats['iterations']) / float(self.iters))
        actor_lr, critic_lr = self.actor_lr, self.critic_lr
        if self.decay_lr:
            actor_lr = max(self.actor_lr * (1.0 - float(self.stats['iterations']) / float(self.iters)), self.min_lr)
            critic_lr = max(self.critic_lr * (1.0 - float(self.stats['iterations']) / float(self.iters)), self.min_lr)
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr
            if self.cmdp_mode:
                for param_group in self.cost_optimizer.param_groups:
                    param_group['lr'] = critic_lr
        clip_ratio = self.clip_ratio
        if self.decay_clip_ratio:
            clip_ratio = max(self.clip_ratio * (1.0 - float(self.stats['iterations']) / float(self.iters)), self.min_clip_ratio)
        # update policy and value networks
        epochs = 0
        kl = 0.0
        stop_actor_training = False
        if self.is_rucur:
            obs_buf, actions_buf, values_buf, advantages_buf, log_probs_buf, cost_values_buf, cost_advantages_buf = (self.obs_buf, self.actions_buf, self.values_buf, self.advantages_buf, self.log_probs_buf, self.cost_values_buf, self.cost_advantages_buf)
            pass
        else:
            # flatten the (N_envs, nsteps, ...) data to (N_envs * nsteps, ...)
            obs_buf, actions_buf, values_buf, advantages_buf, log_probs_buf, cost_values_buf, cost_advantages_buf = (self.obs_buf.reshape(-1, *self.obs_buf.shape[2:]), self.actions_buf.reshape(-1, *self.actions_buf.shape[2:]), self.values_buf.reshape(-1, 1), self.advantages_buf.reshape(-1, 1), self.log_probs_buf.reshape(-1, 1), self.cost_values_buf.reshape(-1, 1), self.cost_advantages_buf.reshape(-1, 1))
        for epoch in range(self.nepochs):
            if self.verbose:
                print(f"Training epoch {epoch + 1}/{self.nepochs}", end='\r')
            self.actor.train()
            self.critic.train()
            if self.cmdp_mode:
                self.cost_critic.train()
            total = self.nsteps * self.envs.num_envs
            idxs = torch.randperm(total, device=self.training_device)   # for non-RNNs
            if self.is_rucur:
                n_segments_along_time = self.nsteps // self.context_length
                n_trajs_per_batch = self.batch_size // self.context_length

            n_batches = total // self.batch_size
            for mb_num in range(n_batches):
                if not self.is_rucur:
                    start = mb_num * self.batch_size
                    end = min(start + self.batch_size, total)
                    mb_idxs = idxs[start:end]
                    mb_loss, _, _, _, _ = self.loss(obs_buf[mb_idxs], actions_buf[mb_idxs], values_buf[mb_idxs], advantages_buf[mb_idxs], log_probs_buf[mb_idxs], cost_values_buf[mb_idxs], cost_advantages_buf[mb_idxs], clip_ratio, float(not stop_actor_training), self.value_loss_coef, self.value_loss_coef * float(self.cmdp_mode), entropy_coef)
                else:
                    n_segments_along_time = self.nsteps // self.context_length
                    n_trajs_per_batch = self.batch_size // self.context_length
                    n_segments_along_batch = obs_buf.shape[0] // n_trajs_per_batch
                    assert n_batches == n_segments_along_batch * n_segments_along_time, "Batch size not compatible with context length and number of environments."
                    # we are moving along a grid of n_segments_along_batch x n_segments_along_time. Scanning along time axis. Whehever time axis index becomes 0, we reset the cache to initial context for the corresponding trajectories along the batch axis.
                    grid_row_index = (mb_num // n_segments_along_time)
                    grid_col_index = (mb_num % n_segments_along_time)
                    dim0_start = grid_row_index * n_trajs_per_batch
                    dim0_end = min(dim0_start + n_trajs_per_batch, obs_buf.shape[0])
                    dim1_start = grid_col_index * self.context_length
                    dim1_end = min(dim1_start + self.context_length, obs_buf.shape[1])
                    dim0_range = list(range(dim0_start, dim0_end))
                    if grid_col_index == 0:
                        self.actor.cache = copy.deepcopy(cache_move_to_device(cache_slice_for_batch_ids(self.actor_inference_context_initial, dim0_range), self.training_device))
                        self.critic.cache = copy.deepcopy(cache_slice_for_batch_ids(self.critic_context_initial, dim0_range))
                        if self.cmdp_mode:
                            self.cost_critic.cache = copy.deepcopy(cache_slice_for_batch_ids(self.cost_critic_context_initial, dim0_range))
                    else:
                        # detach the cache from previous segment to avoid backprop through time
                        self.actor.cache = cache_detach(self.actor.cache)
                        self.critic.cache = cache_detach(self.critic.cache)
                        if self.cmdp_mode:
                            self.cost_critic.cache = cache_detach(self.cost_critic.cache)
                    mb_loss, _, _, _, _ = self.loss(obs_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], actions_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], values_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], advantages_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], log_probs_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], cost_values_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], cost_advantages_buf[dim0_start:dim0_end, dim1_start:dim1_end, ...], clip_ratio, float(not stop_actor_training), self.value_loss_coef, self.value_loss_coef * float(self.cmdp_mode), entropy_coef)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if self.cmdp_mode:
                    self.cost_optimizer.zero_grad()
                mb_loss.backward()
                # update actor
                if not stop_actor_training:
                    # clipnorm (only) if actor is being updated, as it is an expensive operation
                    if self.clipnorm < np.inf:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clipnorm)
                    self.actor_optimizer.step()
                # update critic
                if self.clipnorm < np.inf:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clipnorm)
                self.critic_optimizer.step()
                # update cost critic
                if self.cmdp_mode:
                    if self.clipnorm < np.inf:
                        torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.clipnorm)
                    self.cost_optimizer.step()
            epochs += 1
            if not stop_actor_training:
                self.actor.eval()
                self.actor.cache = copy.deepcopy(cache_move_to_device(self.actor_inference_context_initial, self.training_device))  # set context for RNNs
                # Early stop if measured KL exceeds 1.5 * target_kl to avoid over-updating.
                with torch.no_grad():
                    kl = self.actor.get_kl_div(obs_buf, actions_buf, log_probs_buf).item()
                    if kl > 1.5 * self.target_kl:
                        stop_actor_training = True
                        if self.early_stop_critic:
                            break

        if self.verbose:
            terminal_width = shutil.get_terminal_size((80, 20)).columns
            print(' ' * terminal_width, end='\r')  # erase the progress log

        # compute loss on all data for logging. This will also forward the caches for RNNs to the end of the data and set it up for next iteration.
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            if self.cmdp_mode:
                self.cost_critic.eval()
            self.actor.cache = copy.deepcopy(cache_move_to_device(self.actor_inference_context_initial, self.training_device))  # set context for RNNs
            self.critic.cache = copy.deepcopy(self.critic_context_initial)  # set context for RNNs
            if self.cmdp_mode:
                self.cost_critic.cache = copy.deepcopy(self.cost_critic_context_initial)  # set context for RNNs
            total_loss, actor_loss, critic_loss, cost_loss, entropy = self.loss(obs_buf, actions_buf, values_buf, advantages_buf, log_probs_buf, cost_values_buf, cost_advantages_buf, clip_ratio, 1, self.value_loss_coef, self.value_loss_coef * float(self.cmdp_mode), entropy_coef)
            value = values_buf.mean().item()
            mean_cost = cost_values_buf.mean().item() if self.cmdp_mode else 0.0
            self.actor.cache = self.actor_inference_context_initial  # reset context before calling get_logprob
            logprobs = self.actor.get_logprob(obs_buf, actions_buf)
            mean_logprob = logprobs.mean().item()

            # record cached contexts for RNNs for next iteration
            self.actor_inference_context_initial = copy.deepcopy(cache_move_to_device(cache_detach(self.actor.cache), self.inference_device))
            self.critic_context_initial = copy.deepcopy(cache_detach(self.critic.cache))
            if self.cmdp_mode:
                self.cost_critic_context_initial = copy.deepcopy(cache_detach(self.cost_critic.cache))

        # update actor for inference, if using a different device for inference
        if self.inference_device != self.training_device:
            self.actor_inference.load_state_dict(self.actor.state_dict())

        # Lagrange update:
        if self.cmdp_mode:
            #   λ <- clip(λ + lr * (mean_cost - cost_threshold), [0, lagrange_max])
            # Switch between discounted vs undiscounted cost via `constrain_undiscounted_cost`.
            self.lagrange = self.lagrange + self.lagrange_lr * (self.moving_average_cost - self.cost_threshold)
            self.lagrange = np.clip(self.lagrange, 0.0, self.lagrange_max)  # avoid too large lagrange multiplier

        # update stats
        self.stats['iterations'] += 1
        self.stats['losses'].append(total_loss.item())
        self.stats['actor_losses'].append(actor_loss.item())
        self.stats['critic_losses'].append(critic_loss.item())
        self.stats['cost_losses'].append(cost_loss.item() if cost_loss is not None else 0.0)
        self.stats['entropies'].append(entropy.item())
        self.stats['kl_divs'].append(kl)
        self.stats['values'].append(value)
        self.stats['cost_values'].append(mean_cost)
        self.stats['logprobs'].append(mean_logprob)
        self.stats['actor_lrs'].append(actor_lr)
        self.stats['critic_lrs'].append(critic_lr)
        self.stats['entropy_coefs'].append(entropy_coef)
        self.stats['clip_ratios'].append(clip_ratio)
        self.stats['lagranges'].append(self.lagrange)

    def train(self, for_iterations):
        """
        Train the PPO agent for the specified number of iterations.

        Performs `for_iterations` calls to `train_one_iteration()`, printing progress
        and key metrics after each. Stops early if total iterations reach `self.iters`.

        Args:
            for_iterations: Number of iterations to train (or fewer if already complete).

        Returns:
            None (updates `self.stats` with training history).
        """
        if self.stats['iterations'] >= self.iters:
            print("Training already complete.")
            return
        for it in range(for_iterations):
            self.train_one_iteration()
            print(
                f"Iteration {self.stats['iterations']}/{self.iters} complete.\n"
                f"  Total timesteps: {self.stats['total_timesteps']}\n"
                f"  Total episodes: {self.stats['total_episodes']}\n"
                f"  Mean (100-window) return: {self.stats['mean_returns'][-1]:.6f}\n"
                f"  Mean (100-window) total cost: {self.stats['mean_total_costs'][-1]:.6f}\n"
                f"  Mean (100-window) discounted cost: {self.stats['mean_discounted_costs'][-1]:.6f}\n"
                f"  Mean (100-window) length: {self.stats['mean_lengths'][-1]:.6f}\n"
                f"  Mean (100-window) success: {self.stats['mean_successes'][-1]:.6f}\n"
                f"  Exp. Moving average cost ({'undiscounted' if self.constrain_undiscounted_cost else 'discounted'}) (alpha = {self.moving_cost_estimate_step_size}): {self.moving_average_cost:.6f}\n"
                f"  Lagrange multiplier: {self.stats['lagranges'][-1]:.6f}\n"
                f"  Total loss: {self.stats['losses'][-1]:.6f}\n"
                f"  Actor loss: {self.stats['actor_losses'][-1]:.6f}\n"
                f"  Critic loss: {self.stats['critic_losses'][-1]:.6f}\n"
                f"  Cost critic loss: {self.stats['cost_losses'][-1]:.6f}\n"
                f"  Entropy: {self.stats['entropies'][-1]:.6f}\n"
                f"  KL: {self.stats['kl_divs'][-1]:.6f}\n"
                f"  Value: {self.stats['values'][-1]:.6f}\n"
                f"  Cost value: {self.stats['cost_values'][-1]:.6f}\n"
                f"  Learning rates - actor: {self.stats['actor_lrs'][-1]:.6f}, critic: {self.stats['critic_lrs'][-1]:.6f}\n"
                f"  Entropy coef: {self.stats['entropy_coefs'][-1]:.6f}\n"
                f"  Clip ratio: {self.stats['clip_ratios'][-1]:.6f}\n"
                + ''.join([f"  Mean (100-window) {key}: {self.stats['info_key_means'][key][-1]:.6f}\n" for key in self.custom_info_keys_to_log_at_episode_end]), flush=True)
            if self.stats['iterations'] >= self.iters:
                print("Training complete.")
                break


class POMDPEnvWrapper(gymnasium.Wrapper):
    """A wrapper to augment a flag indicating start of an episode in the observation space. Useful for RNN-based agents in POMDPs. Usually not required for Atari since it is clear which states are episode starts. TODO: This wrapper will concatenate the previous state, previous action, and previous reward to the current observation.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        original_obs_space = env.observation_space
        if isinstance(original_obs_space, Box):
            low = np.concatenate([original_obs_space.low, np.array([0.0], dtype=original_obs_space.dtype)], axis=0)
            high = np.concatenate([original_obs_space.high, np.array([1.0], dtype=original_obs_space.dtype)], axis=0)
            self.observation_space = Box(low=low, high=high, dtype=original_obs_space.dtype)    # type: ignore
        else:
            raise NotImplementedError("POMDPEnvWrapper only supports Box observation spaces.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_with_flag = np.concatenate([obs, np.array([1.0], dtype=obs.dtype)], axis=0)  # episode start flag set to 1.0
        return obs_with_flag, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_with_flag = np.concatenate([obs, np.array([0.0], dtype=obs.dtype)], axis=0)  # episode start flag set to 0.0
        return obs_with_flag, reward, terminated, truncated, info


suffix = "-default"
# Example usage:
#   python ppo_single_file.py <EnvName> <iters> [model_path]
# If model_path is provided, load actor and evaluate; else train from scratch. For atari, specify env name starting with "ALE/".
if __name__ == "__main__":
    # Command-line argument parsing for environment, iterations, and optional model loading.
    envname = sys.argv[1] if len(sys.argv) > 1 else "ALE/Pong-v5"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print(f"Using environment: {envname}")

    def make_env(env_name: str, **kwargs) -> gymnasium.Env:
        if env_name.startswith("ALE/"):  # handle Atari specially with preprocessing and frame stacking.
            # if it is not -v5, then remove the "ALE/" prefix.
            if not env_name.endswith("-v5"):
                env_name = env_name[4:]
            import ale_py
            gymnasium.register_envs(ale_py)
            env = wrappers.FrameStackObservation(wrappers.AtariPreprocessing(make(env_name, frameskip=1, **kwargs), frame_skip=4), stack_size=4, padding_type="zero")
        else:
            env = make(env_name, **kwargs)
        env = wrappers.TimeLimit(env, max_episode_steps=100000)
        return env

    # one single env for recording videos
    env = make_env(envname, render_mode="rgb_array")
    env = wrappers.RecordVideo(env, f"videos/{envname.replace('/', '-')}{suffix}", name_prefix='train', episode_trigger=lambda x: True)
    action_dim: int = env.action_space.n if isinstance(env.action_space, Discrete) else env.action_space.shape[0]  # type: ignore
    actor_out_dim = action_dim if isinstance(env.action_space, Discrete) else 2 * action_dim    # mean and logstd for continuous actions
    print(f"obs_shape: {env.observation_space.shape}, action_dim: {action_dim}")

    # vectorized env for training and common PPO kwargs (irrespective of atari vs others)
    envs = AsyncVectorEnv([lambda i=i: make_env(envname) for i in range(8)], **autoreset_kwarg_samestep_newgymapi)
    ppo_kwargs = dict(entropy_coef=0.01, nsteps=256, norm_rewards=True, training_device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True)

    # Model && PPO setup: branch for Atari (CNN) vs others (classic control / Mujoco / MLP).
    if envname.startswith("ALE/"):
        actor_model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),      # 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),     # 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),     # 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),                                   # 3136
            nn.Linear(3136, 512),                           # 512
            nn.ReLU(),
            nn.Linear(512, actor_out_dim)                      # Output layer
        )
        critic_model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),      # 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),     # 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),     # 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),                                   # 3136
            nn.Linear(3136, 512),                           # 512
            nn.ReLU(),
            nn.Linear(512, 1)                               # Output layer
        )
        ppo_kwargs.update(dict(actor_lr=0.0001, critic_lr=0.0001, decay_lr=True, decay_entropy_coef=True, batch_size=64, nepochs=4, inference_device=ppo_kwargs['training_device']))
    else:
        obs_dim = env.observation_space.shape[0]    # type: ignore
        actor_model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actor_out_dim)
        )
        critic_model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    actor = Actor(actor_model, env.observation_space, env.action_space, deterministic=False, norm_obs=True, state_dependent_std=True)  # type: ignore
    critic = Critic(critic_model, env.observation_space, norm_obs=True)  # type: ignore
    ppo = PPO(envs, actor, critic, iters, **ppo_kwargs)  # type: ignore

    # Optional: Load a pre-trained model and evaluate.
    test_model = sys.argv[3] if len(sys.argv) > 3 else ""
    if test_model != "":
        assert os.path.exists(test_model), f"Model path {test_model} does not exist"
        print(f"Testing model from {test_model}")
        actor.load_state_dict(torch.load(test_model, map_location=ppo.inference_device))
        rs, _, _, _ = actor.evaluate_policy_parallel(envs, ppo.iters, deterministic=True)  # run parallel tests over iters episodes
        print(f"Mean return over {ppo.iters} episodes: {np.mean(rs):.6f} +/- {np.std(rs):.6f}")
        env.name_prefix = "test"
        actor.evaluate_policy(env, 5, deterministic=True)  # run few episodes and record video
    else:
        print("Training model from scratch")
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        while ppo.stats['iterations'] < ppo.iters:
            ppo.train(100)                                           # train in chunks of 100 iterations and save model, record a video, and plot training curve
            torch.save(ppo.actor.state_dict(), f"models/{envname.replace('/', '-')}{suffix}.pth")
            actor.evaluate_policy(env, 1, deterministic=False)      # for video recording. Keeping deterministic=False to see the exact policy as in training
            plt.clf()
            plt.plot(np.arange(ppo.stats['iterations']) + 1, ppo.stats['mean_returns'])
            plt.gca().set(xlabel="Iteration", ylabel="Mean Return (100-episode window)", title=f"PPO on {envname}")
            plt.savefig(f"plots/{envname.replace('/', '-')}{suffix}.png")

    # Cleanup environments.
    envs.close()
    env.close()
