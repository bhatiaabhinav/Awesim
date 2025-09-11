from stable_baselines3 import DQN

import numpy as np
import torch as th
from torch.nn import functional as F

class DoubleDQN(DQN):
    """
    Double DQN implementation using Stable Baselines3's DQN.
    This class can be extended to implement custom behavior for Double DQN.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        val_fns = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():

                # The following (from DQN) changes:
                # # Compute the next Q-values using the target network
                # next_q_values = self.q_net_target(replay_data.next_observations)
                # # Follow greedy policy: use the one with the highest value
                # next_q_values, _ = next_q_values.max(dim=1)
                # # Avoid potential broadcast issue
                # next_q_values = next_q_values.reshape(-1, 1)

                # DDQN:
                # Compute Q-values for next observations using the online Q-network to select the best action
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1).unsqueeze(1)
                # Compute Q-values for next observations using the target Q-network
                next_q_target = self.q_net_target(replay_data.next_observations)
                # Gather the Q-values corresponding to the selected actions
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute max Q-values to record value function
            val_fn = current_q_values.max(dim=1)[0].mean().item()
            val_fns.append(val_fn)

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/max_q_mean", np.mean(val_fns))
        self.logger.record("train/loss", np.mean(losses))
