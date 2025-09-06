from stable_baselines3.common.callbacks import BaseCallback


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
        self.model.ent_coef = self.start
        self.model.logger.record("train/ent_coef_schedule", self.model.ent_coef)

    def _on_step(self) -> bool:
        # Progress remaining in [1, 0]
        pr = float(getattr(self.model, "_current_progress_remaining", 1.0))

        # Map progress to [1, 0] over the chosen window:
        # if end_fraction > 0, finish decay early; otherwise decay until the end.
        denom = max(1e-8, 1.0 - self.end_fraction)  # avoid div-by-zero
        # progress within the decay window: 1 at start, 0 at end_fraction
        pr_window = max(0.0, min(1.0, (pr - self.end_fraction) / denom))

        new_ent = self.end + (self.start - self.end) * pr_window
        self.model.ent_coef = float(new_ent)

        # Optional: log to TB/W&B
        self.model.logger.record("train/ent_coef_schedule", self.model.ent_coef)
        return True
