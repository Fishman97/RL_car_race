import random
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

from src.agents.DQNAgent import DQNAgent, DQNConfig
from src.env.carRacingEnv import CarRacingEnv
from src.runner import Runner

# Training hyperparameters
EPISODES = 3_000
MAX_STEPS = 700
SEED = 42
SAVE_DIR = "models"
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
MIN_BUFFER_SIZE = 10_000
LEARNING_RATE = 1e-3
GAMMA = 0.98
N_STEP = 3
UPDATE_FREQUENCY = 1
TARGET_UPDATE_INTERVAL = 2_000
BETA_FRAMES = 30_000
EVAL_INTERVAL = 0


if TYPE_CHECKING:
    from threading import Event


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def find_latest_checkpoint(directory: Path, prefix: str) -> Optional[Path]:
    pattern = f"{prefix}_ep_*.pt"
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None

def evaluate_agent(agent: DQNAgent, env: CarRacingEnv, max_steps: int) -> float:
    observation, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.act(observation, explore=False)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    return total_reward

def train(map_path: str, tileset_path: str, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None, stop_event: Optional["Event"] = None) -> None:
    set_seed(SEED)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    current_last_ckpt = find_latest_checkpoint(save_dir, "dqn_last")
    current_best_ckpt = find_latest_checkpoint(save_dir, "dqn_best")

    eval_env: Optional[CarRacingEnv] = None
    if EVAL_INTERVAL > 0:
        eval_env = CarRacingEnv(map_path, tileset_path, render_mode=None, max_steps=MAX_STEPS)

    env = CarRacingEnv(map_path, tileset_path, render_mode=None, max_steps=MAX_STEPS)
    state_shape = getattr(env.observation_space, "shape", None)
    state_size = int(state_shape[0]) if state_shape else 0
    action_size = int(getattr(env.action_space, "n", 1))

    config = DQNConfig(
        gamma=GAMMA,
        n_step=N_STEP,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        min_buffer_size=MIN_BUFFER_SIZE,
        update_frequency=UPDATE_FREQUENCY,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        beta_frames=BETA_FRAMES,
        model_dir=SAVE_DIR,
    )
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=SEED, config=config, model_path=str(current_last_ckpt) if current_last_ckpt else None)

    last_metadata = agent.get_checkpoint_metadata()
    start_episode = 0
    global_step = 0
    last_episode_reward = float("nan")
    if last_metadata:
        try:
            start_episode = int(last_metadata.get("episode", 0))
        except (TypeError, ValueError):
            start_episode = 0
        try:
            global_step = int(last_metadata.get("global_step", 0))
        except (TypeError, ValueError):
            global_step = 0
        try:
            last_episode_reward = float(last_metadata.get("reward", float("nan")))
        except (TypeError, ValueError):
            last_episode_reward = float("nan")

    best_reward = float("-inf")
    if current_best_ckpt and current_best_ckpt.exists():
        try:
            best_checkpoint = torch.load(current_best_ckpt, map_location=agent.device, weights_only=False)
            best_meta = best_checkpoint.get("metadata", {})
            if isinstance(best_meta, dict):
                best_reward_val = best_meta.get("best_reward")
                if best_reward_val is not None:
                    best_reward = float(best_reward_val)
        except Exception:
            pass
    if best_reward == float("-inf") and not np.isnan(last_episode_reward):
        best_reward = last_episode_reward

    last_episode_number = start_episode

    interrupted = False

    total_target_episode = start_episode + EPISODES

    for episode_idx in range(EPISODES):
        actual_episode = start_episode + episode_idx + 1
        last_episode_number = actual_episode
        if stop_event and stop_event.is_set():
            interrupted = True
            break

        observation, _ = env.reset()
        agent.reset()
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []
        steps = 0

        while not done and steps < MAX_STEPS:
            if stop_event and stop_event.is_set():
                interrupted = True
                break

            action = agent.act(observation, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(observation, action, reward, next_obs, done)

            global_step += 1
            if global_step % config.update_frequency == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

            observation = next_obs
            episode_reward += reward
            steps += 1

        if interrupted:
            break

        avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        last_episode_reward = episode_reward
        base_metadata = {
            "episode": actual_episode,
            "reward": episode_reward,
            "buffer_size": len(agent.replay_buffer),
            "beta": agent.beta,
            "global_step": global_step,
            "best_reward": best_reward,
        }

        last_metadata = dict(base_metadata)
        last_metadata["type"] = "last"

        new_last_ckpt = save_dir / f"dqn_last_ep_{actual_episode:06d}.pt"
        if current_last_ckpt and current_last_ckpt.exists() and current_last_ckpt != new_last_ckpt:
            try:
                current_last_ckpt.unlink()
            except OSError:
                pass
        agent.save(str(new_last_ckpt), metadata=last_metadata)
        current_last_ckpt = new_last_ckpt
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_metadata = dict(base_metadata)
            best_metadata["type"] = "best"
            best_metadata["best_reward"] = best_reward
            new_best_ckpt = save_dir / f"dqn_best_ep_{actual_episode:06d}.pt"
            agent.save(str(new_best_ckpt), metadata=best_metadata)
            current_best_ckpt = new_best_ckpt

        metrics = (
            f"Episode {actual_episode}/{total_target_episode} | "
            f"Reward {episode_reward:.1f} | "
            f"Loss {avg_loss:.4f} | "
            f"Buffer {len(agent.replay_buffer)} | "
            f"Beta {agent.beta:.3f}"
        )
        print(metrics)

        if progress_cb:
            progress_cb(
                {
                    "episode": actual_episode,
                    "reward": episode_reward,
                    "loss": avg_loss,
                    "buffer_size": len(agent.replay_buffer),
                    "beta": agent.beta,
                    "best_reward": best_reward,
                    "total_episodes": total_target_episode,
                }
            )

        if eval_env and EVAL_INTERVAL > 0 and (episode_idx + 1) % EVAL_INTERVAL == 0:
            if stop_event and stop_event.is_set():
                interrupted = True
                break
            eval_reward = evaluate_agent(agent, eval_env, MAX_STEPS)
            print(f"Eval reward {eval_reward:.1f} | Best {best_reward:.1f}")

    final_metadata = {
        "episode": last_episode_number,
        "reward": last_episode_reward,
        "buffer_size": len(agent.replay_buffer),
        "beta": agent.beta,
        "global_step": global_step,
        "type": "last",
        "best_reward": best_reward,
    }
    if current_last_ckpt and current_last_ckpt.exists():
        agent.save(str(current_last_ckpt), metadata=final_metadata)
    else:
        fallback_last = save_dir / f"dqn_last_ep_{last_episode_number:06d}.pt"
        agent.save(str(fallback_last), metadata=final_metadata)
        current_last_ckpt = fallback_last

    env.close()
    if eval_env:
        eval_env.close()


def train_with_UI(map_path: str, tileset_path: str) -> None:
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    training_status: Dict[str, int] = {
        "episode": -1,
        "buffer_size": -1,
        "total_episodes": EPISODES,
    }

    runner: Optional[Runner] = None

    def progress_cb(stats: Dict[str, Any]) -> None:
        episode = stats.get("episode")
        if episode is None:
            return
        try:
            training_status["episode"] = int(episode)
        except (TypeError, ValueError):
            pass

        buffer_val = stats.get("buffer_size")
        if buffer_val is not None:
            try:
                training_status["buffer_size"] = int(buffer_val)
            except (TypeError, ValueError):
                pass

        total_val = stats.get("total_episodes")
        if total_val is not None:
            try:
                training_status["total_episodes"] = int(total_val)
            except (TypeError, ValueError):
                pass

        if runner is not None:
            runner.update_training_status(
                training_status["episode"],
                training_status["total_episodes"],
                training_status["buffer_size"],
            )

    train_thread = threading.Thread(
        target=train,
        kwargs={
            "map_path": map_path,
            "tileset_path": tileset_path,
            "stop_event": stop_event,
            "progress_cb": progress_cb,
        },
        daemon=True,
    )
    latest_last_ckpt = find_latest_checkpoint(save_dir, "dqn_last")
    current_model_path = str(latest_last_ckpt) if latest_last_ckpt else None
    current_mtime = 0.0
    if latest_last_ckpt:
        try:
            current_mtime = latest_last_ckpt.stat().st_mtime
        except OSError:
            current_mtime = 0.0

    runner = Runner(
        map_path=map_path,
        tileset_path=tileset_path,
        agent_type="DQN",
        render_mode="human",
        max_steps=MAX_STEPS,
        model_path=current_model_path,
    )
    runner.update_training_status(
        training_status["episode"],
        training_status["total_episodes"],
        training_status["buffer_size"],
    )

    train_thread.start()

    pending_model_path: Optional[str] = None
    pending_model_mtime = 0.0

    try:
        while not stop_event.is_set():
            keep_running = runner.run()

            latest_last_ckpt = find_latest_checkpoint(save_dir, "dqn_last")
            latest_path_str = str(latest_last_ckpt) if latest_last_ckpt else None

            if latest_last_ckpt:
                try:
                    latest_mtime = latest_last_ckpt.stat().st_mtime
                except OSError:
                    latest_mtime = current_mtime
            else:
                latest_mtime = 0.0

            if latest_path_str and (latest_path_str != current_model_path or latest_mtime > current_mtime):
                pending_model_path = latest_path_str
                pending_model_mtime = latest_mtime

            if runner.is_window_closed():
                stop_event.set()
                break

            if not keep_running:
                if pending_model_path is not None:
                    if runner.update_model_path(pending_model_path):
                        current_model_path = pending_model_path
                        current_mtime = pending_model_mtime
                    pending_model_path = None
                if stop_event.is_set():
                    break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        runner.close()
        train_thread.join()
