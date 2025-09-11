import numpy as np
import torch
from typing import Optional

import bindings as A
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def direction_onehot(direction) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    if direction == A.DIRECTION_NORTH:
        onehot[0] = 1.0
    elif direction == A.DIRECTION_EAST:
        onehot[1] = 1.0
    elif direction == A.DIRECTION_SOUTH:
        onehot[2] = 1.0
    elif direction == A.DIRECTION_WEST:
        onehot[3] = 1.0
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return onehot


def indicator_onehot(indicator) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    if indicator == A.INDICATOR_LEFT:
        onehot[0] = 1.0
    elif indicator == A.INDICATOR_NONE:
        onehot[1] = 1.0
    elif indicator == A.INDICATOR_RIGHT:
        onehot[2] = 1.0
    else:
        raise ValueError(f"Unknown indicator: {indicator}")
    return onehot



class EntropyDecayCallback(BaseCallback):
    """
    Linearly decays PPO's entropy coefficient during training.

    Args:
        start (float): initial ent_coef value at progress_remaining=1.0
        end (float): final ent_coef value reached when progress_remaining hits 0
        end_fraction (float): fraction of training (in progress_remaining space)
                              over which to finish the decay, e.g. 0.2 -> done by 80% of timesteps
    """
    def __init__(self, start: float, end: float, end_fraction: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        # progress_remaining goes 1.0 -> 0.0; we finish the decay when it reaches end_fraction
        self.end_fraction = float(end_fraction)

    def _on_training_start(self) -> None:
        # Initialize ent_coef
        if isinstance(self.model, PPO):
            model: PPO = self.model
            model.ent_coef = self.start
            model.logger.record("train/ent_coef_schedule", model.ent_coef)
        else:
            raise ValueError("EntropyDecayCallback only works with PPO")

    def _on_step(self) -> bool:
        # Progress remaining in [1, 0]
        pr = float(getattr(self.model, "_current_progress_remaining", 1.0))

        # Map progress to [1, 0] over the chosen window:
        # if end_fraction > 0, finish decay early; otherwise decay until the end.
        denom = max(1e-8, 1.0 - self.end_fraction)  # avoid div-by-zero
        # progress within the decay window: 1 at start, 0 at end_fraction
        pr_window = max(0.0, min(1.0, (pr - self.end_fraction) / denom))

        new_ent = self.end + (self.start - self.end) * pr_window

        if isinstance(self.model, PPO):
            model: PPO = self.model
            model.ent_coef = float(new_ent)
            model.logger.record("train/ent_coef_schedule", model.ent_coef)
        else:
            raise ValueError("EntropyDecayCallback only works with PPO")
        return True


class MlpWithLayerNorm(BaseFeaturesExtractor):
    """MLP feature extractor with layer normalization after each layer."""

    def __init__(self, observation_space, features_dim: int = 256, net_arch: Optional[list] = None, final_norm_and_act: bool = True):
        super().__init__(observation_space, features_dim)
        if net_arch is None:
            net_arch = [features_dim * 4, features_dim * 2]
        n_input = observation_space.shape[0]
        layers = []
        current_dim = n_input
        for hidden_dim in net_arch:
            layers.extend([
                torch.nn.Linear(current_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
            ])
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, features_dim))
        if final_norm_and_act:
            layers.extend([torch.nn.LayerNorm(features_dim), torch.nn.ReLU()])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
    

POLICY_NET_1024_512_256_RELU_LN_CONFIG = {
    "net_arch": [],
    "features_extractor_class": MlpWithLayerNorm,
    "features_extractor_kwargs": {"net_arch": [1024, 512], "features_dim": 256, "final_norm_and_act": True},
}