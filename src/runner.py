from typing import Any, Dict, Optional
from src.env.carRacingEnv import CarRacingEnv
from src.agents.manualAgent import ManualAgent
from src.agents.DQNAgent import DQNAgent
from src.agents.randomAgent import RandomAgent
import time


class Runner:
    def __init__(self, map_path, tileset_path, agent_type, render_mode, max_steps: int, model_path: Optional[str] = None, demo_mode: bool = False):
        self.map_path = map_path
        self.tileset_path = tileset_path
        self.agent_type = agent_type
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.demo_mode = demo_mode

        self.env = CarRacingEnv(map_path=self.map_path, tileset_path=self.tileset_path, render_mode=self.render_mode, max_steps=self.max_steps)

        self.agent: Optional[Any] = None
        self.runs = 0
        self.display_model_episode = -1
        self.model_path = model_path
        self.window_closed = False
        self.training_episode = -1
        self.training_total_episodes = -1
        self.training_buffer_size = -1
        self.env.update_metadata(demo_mode=self.demo_mode)

        state_shape = getattr(self.env.observation_space, "shape", None)
        state_size = int(state_shape[0]) if state_shape else 0
        action_size = int(getattr(self.env.action_space, "n", 1))

        if self.agent_type == "DQN":
            self.agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, model_path=model_path)
            metadata = self.agent.get_checkpoint_metadata()
            if metadata:
                try:
                    self.display_model_episode = int(metadata.get("episode", -1))
                except (TypeError, ValueError):
                    self.display_model_episode = -1
        elif self.agent_type == "random":
            self.agent = RandomAgent(state_size=state_size, action_size=action_size, seed=0)
        elif self.agent_type == "manual":
            self.agent = ManualAgent()

            print("\n=== MANUAL CONTROL ===")
            print("Discrete actions triggered by arrow keys:")
            print("  ↑ Up Arrow:    Accelerate (action 0)")
            print("  ↓ Down Arrow:  Brake      (action 1)")
            print("  ← Left Arrow:  Steer left (action 2)")
            print("  → Right Arrow: Steer right(action 3)")
            print("Release all keys to coast (no-op).")
            print("Close the window to exit.")
            print("======================\n")
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        self._observation: Optional[Any] = None
        self._info: Dict[str, Any] = {}
        self._step_count = 0
        self._episode_active = False
        self._start_time = 0.0

    def _start_episode(self) -> None:
        obs, info = self.env.reset()
        self._observation = obs
        self._info = info
        self._step_count = 0
        self._episode_active = True
        self._start_time = time.time()
        self.window_closed = False
        if self.agent is not None and hasattr(self.agent, "reset"):
            self.agent.reset()
        display_value = self.display_model_episode if self.display_model_episode >= 0 else self.runs + 1
        self.env.update_metadata(display_run=display_value, training_episode=self.training_episode, total_episodes=self.training_total_episodes, buffer_size=self.training_buffer_size, demo_mode=self.demo_mode)

    def run(self, print_interval: int = 60) -> bool:
        if self.agent is None:
            raise RuntimeError("Agent not initialized.")
        if not self._episode_active:
            self._start_episode()
        if self._observation is None:
            return False

        action = self.agent.act(self._observation)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._observation = observation
        self._info = info
        self._step_count += 1

        display_value = self.display_model_episode if self.display_model_episode >= 0 else self.runs + 1
        self.env.update_metadata(display_run=display_value, training_episode=self.training_episode, total_episodes=self.training_total_episodes, buffer_size=self.training_buffer_size, demo_mode=self.demo_mode)

        if print_interval > 0 and self._step_count % print_interval == 0:
            elapsed_time = time.time() - self._start_time
            sps = self._step_count / elapsed_time if elapsed_time > 0 else 0.0
            print(f"\nStep {self._step_count}:")
            print(f"  Position: {info['position']}")
            print(f"  Velocity: {info['velocity']:.2f}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Steps/sec: {sps:.2f}")
            print(f"  Observation: {observation}")

        if not self.env.render():
            self.window_closed = True
            self._episode_active = False
            self.env.close()
            print("\nEnvironment closed.")
            return False

        episode_done = terminated or truncated or self._step_count >= self.max_steps
        if episode_done:
            self.runs += 1
            print(f"\n=== RUN {self.runs} ENDED at step {self._step_count} ===")
            print(f"  Collision: {info['collision']}")
            print(f"  Finished: {info['finished']}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            self._episode_active = False
            return False

        return True

    def update_model_path(self, model_path: Optional[str]) -> bool:
        if self.agent_type != "DQN" or self.agent is None:
            return False
        if model_path == self.model_path:
            return False

        previous_path = self.model_path
        previous_display = self.display_model_episode

        self.model_path = model_path
        if model_path:
            try:
                self.agent.load(model_path)
            except (OSError, RuntimeError) as exc:
                print(f"Warning: failed to load checkpoint '{model_path}': {exc}")
                self.model_path = previous_path
                self.display_model_episode = previous_display
                return False

            metadata = self.agent.get_checkpoint_metadata()
            if metadata:
                try:
                    self.display_model_episode = int(metadata.get("episode", -1))
                except (TypeError, ValueError):
                    self.display_model_episode = -1
            else:
                self.display_model_episode = -1
        else:
            self.display_model_episode = -1

        if self.agent is not None and hasattr(self.agent, "reset"):
            self.agent.reset()
        self._episode_active = False
        if self.env is not None:
            display_value = self.display_model_episode if self.display_model_episode >= 0 else self.runs + 1
            self.env.update_metadata(display_run=display_value, training_episode=self.training_episode, total_episodes=self.training_total_episodes, buffer_size=self.training_buffer_size, demo_mode=self.demo_mode)
        return True

    def is_window_closed(self) -> bool:
        return self.window_closed

    def update_training_status(self, episode: int, total_episodes: int, buffer_size: int) -> None:
        self.training_episode = episode
        self.training_total_episodes = total_episodes
        self.training_buffer_size = buffer_size
        if self._episode_active and self.env is not None:
            display_value = self.display_model_episode if self.display_model_episode >= 0 else self.runs + 1
            self.env.update_metadata(display_run=display_value, training_episode=self.training_episode, total_episodes=self.training_total_episodes, buffer_size=self.training_buffer_size, demo_mode=self.demo_mode)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.window_closed = True
        self._episode_active = False
        self._observation = None
        self.training_episode = -1
        self.training_total_episodes = -1
        self.training_buffer_size = -1
