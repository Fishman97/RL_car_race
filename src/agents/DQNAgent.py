import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 3
    buffer_size: int = 200_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_buffer_size: int = 5_000
    update_frequency: int = 1
    target_update_interval: int = 2_000
    num_atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 200_000
    prior_eps: float = 1e-6
    grad_clip: float = 10.0
    model_dir: str = "models"


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init: float) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = torch.outer(epsilon_out, epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingC51Network(nn.Module):
    def __init__(self, state_size: int, action_size: int, num_atoms: int):
        super().__init__()
        hidden_size = 16
        self.action_size = action_size
        self.num_atoms = num_atoms

        self.fc1 = NoisyLinear(state_size, hidden_size)
        self.fc2 = NoisyLinear(hidden_size, hidden_size)
        self.fc3 = NoisyLinear(hidden_size, hidden_size)

        self.value = NoisyLinear(hidden_size, num_atoms)
        self.advantage = NoisyLinear(hidden_size, action_size * num_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage(x).view(-1, self.action_size, self.num_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=2)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    def __init__(self, state_size: int, capacity: int, alpha: float, prior_eps: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.prior_eps = prior_eps

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        priorities = self.priorities[: self.size]
        scaled_priorities = np.power(priorities + self.prior_eps, self.alpha)
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = np.power(self.size * probs[indices], -beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self.prior_eps
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, seed: int, model_path: Optional[str] = None, config: Optional[DQNConfig] = None) -> None:
        if state_size <= 0:
            raise ValueError("state_size must be positive for DQNAgent.")
        if action_size <= 0:
            raise ValueError("action_size must be positive for DQNAgent.")

        self.state_size = state_size
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)

        self.device = "cpu" #Not tested with CUDA, the bottle neck will probably be the Evn anyway as the NN is small
        self.config = config or DQNConfig()

        torch.manual_seed(seed)

        self.online_net = DuelingC51Network(state_size, action_size, self.config.num_atoms).to(self.device)
        self.target_net = DuelingC51Network(state_size, action_size, self.config.num_atoms).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(
            state_size=state_size,
            capacity=self.config.buffer_size,
            alpha=self.config.alpha,
            prior_eps=self.config.prior_eps,
        )

        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.config.n_step)

        self.gamma = self.config.gamma
        self.gamma_n = self.gamma ** self.config.n_step

        self.support = torch.linspace(self.config.v_min, self.config.v_max, self.config.num_atoms, device=self.device)
        self.delta_z = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)

        self.beta = self.config.beta_start
        self.beta_increment = (1.0 - self.config.beta_start) / max(1, self.config.beta_frames)

        self.learn_step = 0
        self.total_steps = 0

        self.model_path = model_path
        self.last_checkpoint_metadata: Dict[str, Any] = {}
        if model_path:
            self.load(model_path)

    def act(self, observation: np.ndarray, explore: bool = False) -> int:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        if explore:
            self.online_net.train()
            self.online_net.reset_noise()
        else:
            self.online_net.eval()

        with torch.no_grad():
            dist = self.online_net(state)
            q_values = torch.sum(dist * self.support.view(1, 1, -1), dim=2)
            action = torch.argmax(q_values, dim=1)

        return int(action.item())

    def reset(self) -> None:
        self.online_net.eval()

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.config.n_step and not done:
            return

        reward_n, next_state_n, done_n = self._get_n_step_info()
        state_0, action_0 = self.n_step_buffer[0][:2]
        self.replay_buffer.add(state_0, action_0, reward_n, next_state_n, done_n)

        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                reward_n, next_state_n, done_n = self._get_n_step_info()
                state_0, action_0 = self.n_step_buffer[0][:2]
                self.replay_buffer.add(state_0, action_0, reward_n, next_state_n, done_n)
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.popleft()

    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        reward, next_state, done_flag = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, next_s, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                next_state = next_s
                done_flag = True
                break
        return reward, next_state, done_flag

    def can_learn(self) -> bool:
        return len(self.replay_buffer) >= self.config.min_buffer_size

    def learn(self) -> Optional[float]:
        if not self.can_learn():
            return None
        self.learn_step += 1
        self.total_steps += 1

        batch_size = min(self.config.batch_size, len(self.replay_buffer))

        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(
            batch_size,
            self.beta,
        )

        self.beta = min(1.0, self.beta + self.beta_increment)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        self.online_net.train()
        self.target_net.train()
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        dist = self.online_net(states_t)
        action_dist = dist[torch.arange(batch_size, device=self.device), actions_t]
        action_dist = action_dist.clamp(min=1e-6)
        log_prob = torch.log(action_dist)

        with torch.no_grad():
            next_dist_online = self.online_net(next_states_t)
            next_q = torch.sum(next_dist_online * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(next_q, dim=1)

            target_dist = self.target_net(next_states_t)
            next_action_dist = target_dist[torch.arange(batch_size, device=self.device), next_actions]

            Tz = rewards_t.unsqueeze(1) + (1.0 - dones_t.unsqueeze(1)) * self.gamma_n * self.support.view(1, -1)
            Tz = Tz.clamp(self.config.v_min, self.config.v_max)
            b = (Tz - self.config.v_min) / self.delta_z
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l = l.clamp(0, self.config.num_atoms - 1)
            u = u.clamp(0, self.config.num_atoms - 1)

            batch_range = torch.arange(batch_size, device=self.device)
            m = torch.zeros(batch_size, self.config.num_atoms, device=self.device)
            for atom in range(self.config.num_atoms):
                l_idx = l[:, atom]
                u_idx = u[:, atom]
                prob = next_action_dist[:, atom]

                eq_mask = l_idx == u_idx
                m[batch_range[eq_mask], l_idx[eq_mask]] += prob[eq_mask]

                ne_mask = l_idx != u_idx
                if ne_mask.any():
                    batch_ne = batch_range[ne_mask]
                    l_idx_ne = l_idx[ne_mask]
                    u_idx_ne = u_idx[ne_mask]
                    prob_ne = prob[ne_mask]
                    m[batch_ne, l_idx_ne] += prob_ne * (u[ne_mask, atom].float() - b[ne_mask, atom])
                    m[batch_ne, u_idx_ne] += prob_ne * (b[ne_mask, atom] - l[ne_mask, atom].float())

            m = m + 1e-6
            m = m / m.sum(dim=1, keepdim=True)

        loss = -(m * log_prob).sum(dim=1)
        loss = (loss * weights_t).mean()

        if torch.isnan(loss):
            return None

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            q_online = torch.sum(action_dist * self.support.view(1, -1), dim=1)
            q_target = torch.sum(m * self.support.view(1, -1), dim=1)
            td_error = (q_online - q_target).abs().clamp(min=self.config.prior_eps)

        self.replay_buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()

        self.online_net.eval()
        self.target_net.eval()
        return float(loss.item())

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        metadata_dict: Dict[str, Any] = dict(metadata) if metadata else {}
        self.last_checkpoint_metadata = metadata_dict
        checkpoint: Dict[str, object] = {
            "online_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "learn_step": self.learn_step,
            "total_steps": self.total_steps,
            "metadata": metadata_dict,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint.get("online_state_dict", {}))
        self.target_net.load_state_dict(self.online_net.state_dict())
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError:
                pass
        self.learn_step = int(checkpoint.get("learn_step", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        raw_metadata = checkpoint.get("metadata", {})
        if isinstance(raw_metadata, dict):
            self.last_checkpoint_metadata = dict(raw_metadata)
        else:
            self.last_checkpoint_metadata = {}
        self.online_net.eval()
        self.target_net.eval()

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return dict(self.last_checkpoint_metadata)