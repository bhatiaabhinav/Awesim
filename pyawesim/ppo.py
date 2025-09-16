import copy
import shutil
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import Env, make
from gymnasium.spaces import Box
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv
from torch import Tensor, nn


def log_normal_prob(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Compute log probability of x under a normal distribution with given mean and std."""
    var = std ** 2
    log_prob = -0.5 * (((x - mean) ** 2) / var + 2 * torch.log(std + 1e-6) + np.log(2 * np.pi))
    return log_prob.sum(axis=-1, keepdim=True)


def gaussian_entropy(logstd: Tensor) -> Tensor:
    """Compute the entropy of a Gaussian distribution given its log standard deviation."""
    entropy = torch.sum(logstd + 0.5 * (1.0 + np.log(2 * np.pi)), dim=-1)
    return entropy


class Actor(nn.Module):
    def __init__(self, mean_model, action_space: Box, deterministic: bool = False, device='cpu', norm_obs: bool = False):
        super(Actor, self).__init__()
        assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high))
        self.mean_model = mean_model.to(device)
        self.logstd = nn.Parameter(torch.zeros(action_space.shape[0], dtype=torch.float32, device=device), requires_grad=True)
        self.shift = nn.Parameter(torch.tensor(action_space.low + action_space.high, dtype=torch.float32, device=device) / 2.0, requires_grad=False)
        self.scale = nn.Parameter(torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device), requires_grad=False)
        self.deterministic = deterministic
        self.norm_obs = norm_obs
        if self.norm_obs:
            obs_dim = self.mean_model[0].in_features
            self.obs_mean = nn.Parameter(torch.zeros(obs_dim, dtype=torch.float32, device=device), requires_grad=False)
            self.obs_var = nn.Parameter(torch.ones(obs_dim, dtype=torch.float32, device=device), requires_grad=False)
            self.obs_count = nn.Parameter(torch.tensor(0., dtype=torch.float32, device=device), requires_grad=False)

    def get_mean_logstd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.norm_obs and self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            delta = batch_mean - self.obs_mean
            tot_count = self.obs_count + batch_count
            new_mean = self.obs_mean + delta * batch_count / tot_count
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + torch.square(delta) * (self.obs_count * batch_count / tot_count)
            new_var = m_2 / tot_count
            new_count = tot_count
            self.obs_mean.data.copy_(new_mean)
            self.obs_var.data.copy_(new_var)
            self.obs_count.data.copy_(new_count)
        if self.norm_obs:
            x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        mean = self.mean_model(x)
        logstd = self.logstd
        if self.deterministic:
            logstd = logstd - torch.inf
        logstd = torch.clamp(logstd, -20, 2)  # to avoid numerical issues
        return mean, logstd

    def sample_action_logstd_logprob(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, logstd = self.get_mean_logstd(x)
        std = torch.exp(logstd)
        noise = torch.randn_like(mean)
        action = mean + std * noise
        log_prob = log_normal_prob(action, mean, std)
        action = torch.tanh(action)
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        action = action * self.scale + self.shift
        return action, logstd, log_prob

    def get_logstd_logprob(self, x: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        mean, logstd = self.get_mean_logstd(x)
        std = torch.exp(logstd)
        # Rescale action to [-1, 1]
        action_unshifted_unscaled = (action - self.shift) / (self.scale + 1e-6)
        action_unshifted_unscaled = torch.clamp(action_unshifted_unscaled, -0.999, 0.999)  # otherwise atanh(1) is inf
        action_untanhed = torch.atanh(action_unshifted_unscaled)
        log_prob = log_normal_prob(action_untanhed, mean, std)
        log_prob -= torch.sum(torch.log(1 - action_unshifted_unscaled.pow(2) + 1e-6), dim=-1, keepdim=True)
        return logstd, log_prob

    def get_entropy(self, x):
        _, logstd = self.get_mean_logstd(x)
        return gaussian_entropy(logstd)

    def get_kl_div(self, obs, actions, old_logprobs):
        _, log_probs = self.get_logstd_logprob(obs, actions)
        log_ratio = log_probs - old_logprobs
        kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio)
        # kl = torch.mean(old_logprobs - log_probs)
        return kl

    def get_loss_and_entropy(self, obs, actions, advantages, old_logprobs, clip_ratio):
        _, log_probs = self.get_logstd_logprob(obs, actions)
        ratio = torch.exp(log_probs - old_logprobs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        entropy = gaussian_entropy(self.logstd)
        return policy_loss, entropy

    def forward(self, x):
        # if input is numpy array, convert to torch tensor
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.mean_model.parameters()).device)
        # if input is not batched, add a batch dimension
        input_is_unbatched = len(x.shape) == 1
        if input_is_unbatched:
            x = x.unsqueeze(0)

        # get action
        action, _, _ = self.sample_action_logstd_logprob(x)

        # if input was unbatched, remove the batch dimension
        if input_is_unbatched:
            action = action.squeeze(0)
        # if input was numpy, convert output to numpy
        if input_is_numpy:
            action = action.detach().cpu().numpy()
        return action

    def evaluate_policy(self, env: Env, num_episodes=100, deterministic=True) -> Tuple[List[float], List[int], List[bool]]:
        self.eval()
        self.deterministic = deterministic
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        for episode in range(num_episodes):
            print(f"Evaluating episode {episode + 1}/{num_episodes}", end='\r')
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            length = 0
            while not done:
                action = self(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward  # type: ignore
                length += 1
                if done:
                    episode_successes.append(info.get("is_success", False))
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
        print()
        self.deterministic = False  # undo

        return episode_rewards, episode_lengths, episode_successes


class Critic(nn.Module):
    def __init__(self, model, device='cpu', norm_obs: bool = False):
        super(Critic, self).__init__()
        self.model = model
        self.norm_obs = norm_obs
        if self.norm_obs:
            obs_dim = self.model[0].in_features
            self.obs_mean = nn.Parameter(torch.zeros(obs_dim, dtype=torch.float32, device=device), requires_grad=False)
            self.obs_var = nn.Parameter(torch.ones(obs_dim, dtype=torch.float32, device=device), requires_grad=False)
            self.obs_count = nn.Parameter(torch.tensor(0., dtype=torch.float32, device=device), requires_grad=False)
        self.to(device)

    def get_loss(self, states, old_values, advantages):
        values = (old_values + advantages).detach()
        values_pred = self.get_value(states)
        return F.mse_loss(values_pred, values)

    def get_value(self, x: Tensor) -> Tensor:
        if self.norm_obs and self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            delta = batch_mean - self.obs_mean
            tot_count = self.obs_count + batch_count
            new_mean = self.obs_mean + delta * batch_count / tot_count
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + torch.square(delta) * (self.obs_count * batch_count / tot_count)
            new_var = m_2 / tot_count
            new_count = tot_count
            self.obs_mean.data.copy_(new_mean)
            self.obs_var.data.copy_(new_var)
            self.obs_count.data.copy_(new_count)
        if self.norm_obs:
            x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return self.model(x)

    def forward(self, x):

        # if input is numpy array, convert to torch tensor
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.model.parameters()).device)
        # if input is not batched, add a batch dimension
        input_is_unbatched = len(x.shape) == 1
        if input_is_unbatched:
            x = x.unsqueeze(0)

        # get value
        value = self.get_value(x)

        # if input was unbatched, remove the batch dimension
        if input_is_unbatched:
            value = value.squeeze(0)
        # if input was numpy, convert output to numpy
        if input_is_numpy:
            value = value.detach().cpu().numpy()
        return value


class PPO:
    def __init__(self,
                 envs: VectorEnv,
                 actor: Actor,
                 critic: Critic,
                 iters: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 decay_lr: bool = False,
                 min_lr: float = 1.25e-5,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 nsteps: int = 2048,
                 nepochs: int = 10,
                 batch_size: int = 64,
                 clip_ratio: float = 0.2,
                 target_kl: float = np.inf,
                 early_stop_critic: bool = False,
                 entropy_coef: float = 0.0,
                 decay_entropy_coef: bool = False,
                 normalize_advantages: bool = True,
                 clipnorm: float = 0.5,
                 norm_rewards: bool = False,
                 adam_weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-7,
                 inference_device: str = 'cpu',
                 training_device: str = 'cpu'):

        assert envs.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP, "VectorEnv must be in SAME_STEP autoreset mode"

        self.envs = envs
        self.actor = actor.to(training_device)
        self.critic = critic.to(training_device)
        self.iters = iters
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=adam_weight_decay, eps=adam_epsilon)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=adam_weight_decay, eps=adam_epsilon)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.decay_lr = decay_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.early_stop_critic = early_stop_critic
        self.entropy_coef = entropy_coef
        self.decay_entropy_coef = decay_entropy_coef
        self.normalize_advantages = normalize_advantages
        self.clipnorm = clipnorm
        self.norm_rewards = norm_rewards
        self.inference_device = inference_device
        self.training_device = training_device

        # stats
        self.stats = {
            'iterations': 0,
            'total_timesteps': 0,
            'total_episodes': 0,
            'losses': [],               # after each iteration
            'actor_losses': [],         # after each iteration
            'critic_losses': [],        # after each iteration
            'entropies': [],            # after each iteration
            'kl_divs': [],              # after each iteration
            'values': [],               # mean value of states in the batch after each iteration
            'logprobs': [],             # mean logprob of actions in the batch after each iteration (its negative is the true mean entropy of the actions in the batch)
            'logstds': [],              # mean logstd of actions in the batch after each iteration
            'actor_lrs': [],            # adjusted per iteration
            'critic_lrs': [],           # adjusted per iteration
            'entropy_coefs': [],        # adjusted per iteration
            'returns': [],              # return of each trajectory collected so far
            'discounted_returns': [],   # discounted return of each trajectory collected so far
            'lengths': [],              # length of each trajectory collected so far
            'successes': [],            # success of each trajectory collected so far (if env provides this info)
            'mean_returns': [],         # mean return of latest 100 trajectories after each iteration
            'mean_discounted_returns': [],  # mean discounted return of latest 100 trajectories after each iteration
            'mean_lengths': [],         # mean length of latest 100 trajectories after each iteration
            'mean_successes': [],       # mean success of latest 100 trajectories after each iteration (if env provides this info)
        }

        # data buffers
        state_dim = envs.single_observation_space.shape[0]  # type: ignore
        action_dim = envs.single_action_space.shape[0]  # type: ignore
        M, N = self.nsteps, self.envs.num_envs
        self.obs_buf = torch.zeros((N, M, state_dim), dtype=torch.float32, device=self.training_device)
        self.obs_next_buf = torch.zeros((N, M, state_dim), dtype=torch.float32, device=self.training_device)
        self.actions_buf = torch.zeros((N, M, action_dim), dtype=torch.float32, device=self.training_device)
        self.log_probs_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.rewards_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.terminateds_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.truncateds_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.values_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)
        self.advantages_buf = torch.zeros((N, M, 1), dtype=torch.float32, device=self.training_device)

        # data buffers for collecting trajectories, if using a different device for inference
        if self.inference_device != self.training_device:
            self.obs_buf_inference = torch.zeros((N, M, state_dim), dtype=torch.float32, device=self.inference_device)
            self.obs_next_buf_inference = torch.zeros((N, M, state_dim), dtype=torch.float32, device=self.inference_device)
            self.actions_buf_inference = torch.zeros((N, M, action_dim), dtype=torch.float32, device=self.inference_device)
            self.log_probs_buf_inference = torch.zeros((N, M, 1), dtype=torch.float32, device=self.inference_device)
            self.rewards_buf_inference = torch.zeros((N, M, 1), dtype=torch.float32, device=self.inference_device)
            self.terminateds_buf_inference = torch.zeros((N, M, 1), dtype=torch.float32, device=self.inference_device)
            self.truncateds_buf_inference = torch.zeros((N, M, 1), dtype=torch.float32, device=self.inference_device)
        else:
            self.obs_buf_inference = self.obs_buf
            self.obs_next_buf_inference = self.obs_next_buf
            self.actions_buf_inference = self.actions_buf
            self.log_probs_buf_inference = self.log_probs_buf
            self.rewards_buf_inference = self.rewards_buf
            self.terminateds_buf_inference = self.terminateds_buf
            self.truncateds_buf_inference = self.truncateds_buf

        self.returns_buffer = np.zeros((N,), dtype=np.float32)
        self.discounted_returns_buffer = np.zeros((N,), dtype=np.float32)
        self.lengths_buffer = np.zeros((N,), dtype=np.int32)

        # actor for inference, if using a different device for inference
        if self.inference_device != self.training_device:
            self.actor_inference = copy.deepcopy(self.actor).to(self.inference_device)
        else:
            self.actor_inference = self.actor

        # reset all envs and store initial observation
        self.obs: torch.Tensor = torch.tensor(envs.reset()[0], dtype=torch.float32, device=self.inference_device)

    def collect_trajectories(self) -> None:
        # collect trajectories
        self.actor_inference.eval()
        for t in range(self.nsteps):
            with torch.no_grad():
                # act and step
                a_t, _, logp_t = self.actor_inference.sample_action_logstd_logprob(self.obs)
                obs_next_potentially_resetted, r_t, terminated_t, truncated_t, infos = self.envs.step(a_t.detach().cpu().numpy())
                obs_next = obs_next_potentially_resetted.copy()
                for i in range(self.envs.num_envs):
                    if truncated_t[i]:
                        obs_next[i, :] = infos['final_obs'][i]  # type: ignore

                # store data
                self.obs_buf_inference[:, t, :] = self.obs
                self.obs_next_buf_inference[:, t, :] = torch.tensor(obs_next, dtype=torch.float32, device=self.inference_device)
                self.actions_buf_inference[:, t, :] = a_t
                self.log_probs_buf_inference[:, t, :] = logp_t
                self.rewards_buf_inference[:, t, 0] = torch.tensor(r_t, dtype=torch.float32, device=self.inference_device)
                self.terminateds_buf_inference[:, t, 0] = torch.tensor(terminated_t, dtype=torch.float32, device=self.inference_device)
                self.truncateds_buf_inference[:, t, 0] = torch.tensor(truncated_t, dtype=torch.float32, device=self.inference_device)

                # update current obs
                self.obs.data.copy_(torch.tensor(obs_next_potentially_resetted, device=self.inference_device), non_blocking=True)

                # update returns and lengths
                self.discounted_returns_buffer += r_t * np.power(self.gamma, self.lengths_buffer)
                self.returns_buffer += r_t
                self.lengths_buffer += 1
                for i in range(self.envs.num_envs):
                    if terminated_t[i] or truncated_t[i]:
                        self.stats['returns'].append(self.returns_buffer[i])
                        self.stats['discounted_returns'].append(self.discounted_returns_buffer[i])
                        self.stats['lengths'].append(self.lengths_buffer[i])
                        if 'is_success' in infos['final_info']:
                            self.stats['successes'].append(infos['final_info']['is_success'][i])  # type: ignore
                        self.returns_buffer[i] = 0.0
                        self.discounted_returns_buffer[i] = 0.0
                        self.lengths_buffer[i] = 0
                        self.stats['total_episodes'] += 1

                print(f"Collected steps {(t + 1) * self.envs.num_envs}/{self.nsteps * self.envs.num_envs}", end='\r')
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        print(' ' * terminal_width, end='\r')  # erase the progress log

        # if using different device for inference, copy data to training device
        if self.inference_device != self.training_device:
            self.obs_buf.data.copy_(self.obs_buf_inference.data, non_blocking=True)
            self.obs_next_buf.data.copy_(self.obs_next_buf_inference.data, non_blocking=True)
            self.actions_buf.data.copy_(self.actions_buf_inference.data, non_blocking=True)
            self.log_probs_buf.data.copy_(self.log_probs_buf_inference.data, non_blocking=True)
            self.rewards_buf.data.copy_(self.rewards_buf_inference.data, non_blocking=True)
            self.terminateds_buf.data.copy_(self.terminateds_buf_inference.data, non_blocking=True)
            self.truncateds_buf.data.copy_(self.truncateds_buf_inference.data, non_blocking=True)

        # update stats
        self.stats['total_timesteps'] += self.nsteps * self.envs.num_envs
        mean_return = np.mean(self.stats['returns'][-100:]) if len(self.stats['returns']) > 0 else 0.0
        mean_discounted_return = np.mean(self.stats['discounted_returns'][-100:]) if len(self.stats['discounted_returns']) > 0 else 0.0
        mean_length = np.mean(self.stats['lengths'][-100:]) if len(self.stats['lengths']) > 0 else 0.0
        mean_success = np.mean(self.stats['successes'][-100:]) if len(self.stats['successes']) > 0 else 0.0
        self.stats['mean_returns'].append(mean_return)
        self.stats['mean_discounted_returns'].append(mean_discounted_return)
        self.stats['mean_lengths'].append(mean_length)
        self.stats['mean_successes'].append(mean_success)

    def compute_values_advantages(self) -> None:
        # everything happens on training device here
        self.critic.eval()
        with torch.no_grad():
            v = self.critic(self.obs_buf)   # (N, M, 1)
            v_next = self.critic(self.obs_next_buf)  # (N, M, 1)
            advantages = self.rewards_buf + self.gamma * (1.0 - self.terminateds_buf) * v_next - v  # (N, M, 1)
            dones = (self.terminateds_buf + self.truncateds_buf).clamp(max=1.0)
            adv_next = torch.zeros((self.envs.num_envs, 1), dtype=torch.float32, device=self.training_device)
            for t in reversed(range(self.nsteps)):
                advantages[:, t, :] += (1.0 - dones[:, t, :]) * self.gamma * self.lam * adv_next
                adv_next = advantages[:, t, :]

            self.values_buf.data.copy_(v.data, non_blocking=True)
            self.advantages_buf.data.copy_(advantages.data, non_blocking=True)

    def loss(self, data: Optional[Tuple[torch.Tensor, ...]], actor_coeff=1.0, critic_coeff=0.5, entropy_coeff=0.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # data should contain obs, actions, values, advantages, old_logprobs and should be on training device. If none, loss will be computed for all data in buffers
        if data is None:
            obs, actions, old_values, advantages, old_logprobs = self.obs_buf, self.actions_buf, self.values_buf, self.advantages_buf, self.log_probs_buf
        else:
            obs, actions, old_values, advantages, old_logprobs = data

        critic_loss = self.critic.get_loss(obs, old_values, advantages)
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean().detach()) / (advantages.std().detach() + 1e-8)
        actor_loss, entropy = self.actor.get_loss_and_entropy(obs, actions, advantages, old_logprobs, self.clip_ratio)
        total_loss = actor_coeff * actor_loss + critic_coeff * critic_loss - entropy_coeff * entropy
        return total_loss, actor_loss, critic_loss, entropy

    def train_one_iteration(self):
        self.collect_trajectories()
        if self.norm_rewards and len(self.stats['discounted_returns']) > 1:
            reward_std = np.std(self.stats['discounted_returns'])
            if reward_std > 1e-8:
                self.rewards_buf /= reward_std  # type: ignore
        self.compute_values_advantages()
        stop_actor_training = False
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
        epochs = 0
        kl = 0.0
        for epoch in range(self.nepochs):
            print(f"Training epoch {epoch + 1}/{self.nepochs}", end='\r')
            self.actor.train()
            self.critic.train()
            total = self.nsteps * self.envs.num_envs
            idxs = torch.randperm(total, device=self.training_device)
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                mb_idxs = idxs[start:end]
                mb_obs = self.obs_buf.reshape(-1, self.obs_buf.shape[-1])[mb_idxs]
                mb_actions = self.actions_buf.reshape(-1, self.actions_buf.shape[-1])[mb_idxs]
                mb_old_values = self.values_buf.reshape(-1, 1)[mb_idxs]
                mb_advantages = self.advantages_buf.reshape(-1, 1)[mb_idxs]
                mb_old_logprobs = self.log_probs_buf.reshape(-1, 1)[mb_idxs]
                data = (mb_obs, mb_actions, mb_old_values, mb_advantages, mb_old_logprobs)
                mb_loss, _, _, _ = self.loss(data, float(not stop_actor_training), 0.5, entropy_coef)
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                mb_loss.backward()
                # update actor
                if not stop_actor_training:
                    if self.clipnorm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clipnorm)
                    self.actor_optimizer.step()
                # update critic
                if self.clipnorm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clipnorm)
                self.critic_optimizer.step()
            epochs += 1

            self.actor.eval()
            kl = self.actor.get_kl_div(self.obs_buf, self.actions_buf, self.log_probs_buf).item()
            if not stop_actor_training and kl > 1.5 * self.target_kl and self.target_kl > 0:
                stop_actor_training = True
                if self.early_stop_critic:
                    break

        terminal_width = shutil.get_terminal_size((80, 20)).columns
        print(' ' * terminal_width, end='\r')  # erase the progress log

        # compute loss on all data for logging
        self.actor.eval()
        self.critic.eval()
        total_loss, actor_loss, critic_loss, entropy = self.loss(None, 1, 0.5, entropy_coef)
        value = self.values_buf.mean().item()
        logstd, logprobs = self.actor.get_logstd_logprob(self.obs_buf, self.actions_buf)
        mean_logstd = logstd.mean().item()
        mean_logprob = logprobs.mean().item()
        # update stats
        self.stats['iterations'] += 1
        self.stats['losses'].append(total_loss.item())
        self.stats['actor_losses'].append(actor_loss.item())
        self.stats['critic_losses'].append(critic_loss.item())
        self.stats['entropies'].append(entropy.item())
        self.stats['kl_divs'].append(kl)
        self.stats['values'].append(value)
        self.stats['logstds'].append(mean_logstd)
        self.stats['logprobs'].append(mean_logprob)
        self.stats['actor_lrs'].append(actor_lr)
        self.stats['critic_lrs'].append(critic_lr)
        self.stats['entropy_coefs'].append(entropy_coef)

        # update actor for inference, if using a different device for inference
        if self.inference_device != self.training_device:
            self.actor_inference.load_state_dict(self.actor.state_dict())

    def train(self, for_iterations):
        if self.stats['iterations'] >= self.iters:
            print("Training already complete.")
            return
        for it in range(for_iterations):
            self.train_one_iteration()
            print(
                f"Iteration {self.stats['iterations']}/{self.iters} complete.\n"
                f"  Total timesteps: {self.stats['total_timesteps']}\n"
                f"  Total episodes: {self.stats['total_episodes']}\n"
                f"  Mean return: {self.stats['mean_returns'][-1]:.6f}\n"
                f"  Mean length: {self.stats['mean_lengths'][-1]:.6f}\n"
                f"  Mean success: {self.stats['mean_successes'][-1]:.6f}\n"
                f"  Total loss: {self.stats['losses'][-1]:.6f}\n"
                f"  Actor loss: {self.stats['actor_losses'][-1]:.6f}\n"
                f"  Critic loss: {self.stats['critic_losses'][-1]:.6f}\n"
                f"  Entropy: {self.stats['entropies'][-1]:.6f}\n"
                f"  KL: {self.stats['kl_divs'][-1]:.6f}\n"
                f"  Value: {self.stats['values'][-1]:.6f}"
            )
            if self.stats['iterations'] >= self.iters:
                print("Training complete.")
                break


if __name__ == "__main__":   # example usage

    envname = sys.argv[1] if len(sys.argv) > 1 else "Pendulum-v1"
    print(f"Using environment: {envname}")
    env = make(envname)
    obs_dim = env.observation_space.shape[0]    # type: ignore
    action_dim = env.action_space.shape[0]      # type: ignore
    assert isinstance(env.observation_space, Box) and isinstance(env.action_space, Box), "Only Box observation and action spaces are supported."
    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}")

    actor_model = nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim)
    )
    critic_model = nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    actor = Actor(actor_model, env.action_space, deterministic=False, norm_obs=True)  # type: ignore
    critic = Critic(critic_model, norm_obs=True)  # type: ignore

    envs = SyncVectorEnv([lambda: make(envname) for _ in range(4)], autoreset_mode=AutoresetMode.SAME_STEP)
    ppo = PPO(envs, actor, critic, 100, nsteps=256, batch_size=64, target_kl=0.01, actor_lr=0.0001, critic_lr=0.0003, inference_device='cpu', training_device='cuda' if torch.cuda.is_available() else 'cpu')

    ppo.train(ppo.iters)
    envs.close()

    import matplotlib.pyplot as plt

    plt.plot(ppo.stats['mean_returns'])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return (over 100-episode moving window)")
    plt.title(f"PPO on {envname}")
    plt.show()
