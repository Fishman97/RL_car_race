import numpy as np


class RandomAgent:
    def __init__(self, state_size: int, action_size: int, seed: int):
        self.state_size = state_size
        self.action_size = max(1, action_size)
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        # Sample a uniformly random discrete action each step
        return int(self.rng.integers(0, self.action_size))

    def reset(self) -> None:
        pass
