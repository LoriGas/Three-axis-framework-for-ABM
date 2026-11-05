"""Neural agent (V8--V14) - MLP-controlled decision making."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import ACTIONS, ModelParams
from network import NeuralNetwork
from .base import AgentBase


def _valid_action_mask_numpy(percept: np.ndarray) -> np.ndarray:
    """True where action is valid (free neighbour or stay)."""
    p = np.asarray(percept, dtype=np.float32)
    single = p.ndim == 1
    if single:
        p = p[np.newaxis, :]
    mask = np.concatenate([p[:, 4:8] < 0.5, np.ones((p.shape[0], 1), dtype=bool)], axis=1)
    return mask[0] if single else mask


def mask_logits_invalid_numpy(logits: np.ndarray, percept: np.ndarray) -> np.ndarray:
    masked = np.array(logits, copy=True)
    mask = _valid_action_mask_numpy(percept)
    move_logits = masked[..., :5]
    move_logits[~mask] = -1e9
    masked[..., :5] = move_logits
    return masked


def mask_logits_invalid_torch(logits: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
    p = percept if percept.dim() > 1 else percept.unsqueeze(0)
    mask = torch.cat([p[:, 4:8] < 0.5, torch.ones((p.size(0), 1), dtype=torch.bool)], dim=1)
    masked = logits.clone()
    masked[:, :5] = masked[:, :5].masked_fill(~mask, -1e9)
    return masked


def _mask_move_logits(move_logits: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
    """Mask invalid movement directions in a 5-logit tensor."""
    p = percept if percept.dim() > 1 else percept.unsqueeze(0)
    mask = torch.cat([p[:, 4:8] < 0.5, torch.ones((p.size(0), 1), dtype=torch.bool)], dim=1)
    return move_logits.clone().masked_fill(~mask, -1e9)


class NeuralAgent(AgentBase):
    """Uses a NeuralNetwork to decide actions via n-step A2C with decomposed rewards."""

    def __init__(
        self,
        x: int,
        y: int,
        energy: float,
        num_layers: int,
        net_template: Optional[NeuralNetwork] = None,
        online_learning: bool = False,
        stochastic_policy: bool = False,
        lr: float = 0.0003,
        entropy_beta: float = 0.01,
        critic_coef: float = 0.5,
        n_steps: int = 4,
        gamma: float = 0.95,
    ) -> None:
        super().__init__(x, y, energy)
        self.net = NeuralNetwork(num_hidden_layers=num_layers)
        self.online_learning = online_learning
        # Online policies sample actions to preserve exploration while learning.
        self.stochastic_policy = bool(stochastic_policy or online_learning)
        self.entropy_beta = entropy_beta
        self.critic_coef = critic_coef
        self._last_action_idx: Optional[int] = None
        self._last_percept: Optional[List[float]] = None
        self._last_repro_action: Optional[int] = None
        self._last_can_reproduce: bool = True
        self._n_steps = n_steps
        self._gamma = gamma
        # each entry: (s, move, reproduce, r_energy, r_fitness, s_next, done)
        self._traj: list = []

        if net_template is not None:
            self.net.copy_from(net_template)
        if self.online_learning:
            self.net.train()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.net.eval()

    def decide_action(self, percept: List[float], params: ModelParams) -> Tuple[int, int, bool]:
        # Whether reproduction is structurally feasible (free adjacent cell exists)
        self._last_can_reproduce = any(percept[4 + i] < 0.5 for i in range(4))

        if self.stochastic_policy:
            inp = torch.tensor(percept, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw = self.net(inp)  # [1, 6] concatenated logits
            logits = mask_logits_invalid_torch(raw.detach(), inp)

            move_probs = torch.softmax(logits[:, :5], dim=-1)
            move_dist = torch.distributions.Categorical(probs=move_probs)
            move_action = move_dist.sample()
            action_idx = int(move_action.item())

            # Reproduction: SAMPLE from Bernoulli so A2C can explore both decisions
            repro_dist = torch.distributions.Bernoulli(logits=raw[0, 5])
            repro_action = repro_dist.sample()
            repro_idx = int(repro_action.item())

            if self.online_learning:
                self._last_action_idx = action_idx
                self._last_repro_action = repro_idx
                self._last_percept = list(percept)
            dx, dy = ACTIONS[action_idx]
            return dx, dy, bool(repro_idx)

        # Inference path (non-learning) — greedy, no gradient needed
        inp = np.array(percept, dtype=np.float32)
        raw = self.net.forward_numpy(inp)
        logits = mask_logits_invalid_numpy(raw, inp)
        move_idx = int(np.argmax(logits[:5]))
        reproduce = bool(1.0 / (1.0 + np.exp(-float(raw[5]))) >= 0.5)
        dx, dy = ACTIONS[move_idx]
        return dx, dy, reproduce

    def update_online(
        self,
        r_energy: float,
        r_fitness: float,
        next_percept: Optional[List[float]] = None,
        done: bool = False,
    ) -> None:
        """Store a transition and perform bootstrapped n-step A2C updates."""
        if not self.online_learning or self._last_percept is None:
            return

        self._traj.append((
            self._last_percept,
            self._last_action_idx,
            self._last_repro_action,
            float(r_energy),
            float(r_fitness),
            list(next_percept) if next_percept is not None else None,
            bool(done),
        ))
        self._last_percept = None
        self._last_action_idx = None
        self._last_repro_action = None

        if done:
            # A terminal partial rollout is still a valid on-policy batch.
            self._do_rollout_update(self._traj)
            self._traj.clear()
        elif len(self._traj) >= self._n_steps:
            rollout = self._traj[:self._n_steps]
            self._do_rollout_update(rollout)
            del self._traj[:self._n_steps]

    def flush_online(self) -> None:
        """Flush complete buffered transitions at a non-terminal time limit."""
        if not self.online_learning:
            return
        if self._traj:
            self._do_rollout_update(self._traj)
            self._traj.clear()

    def _do_rollout_update(self, transitions: list) -> None:
        """Update every state in one non-overlapping, bootstrapped A2C rollout."""
        if not transitions:
            return
        last = transitions[-1]

        # Bootstrap from V(s_{t+n}) only for a non-terminal transition.
        G_energy = 0.0
        G_fitness = 0.0
        if not last[6] and last[5] is not None:
            next_inp = torch.tensor(last[5], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, next_v_move, next_v_repro = self.net.forward_with_all(next_inp)
            G_energy = float(next_v_move.item())
            G_fitness = float(next_v_repro.item())

        energy_returns: List[float] = []
        fitness_returns: List[float] = []
        for entry in reversed(transitions):
            G_energy = entry[3] + self._gamma * G_energy
            G_fitness = entry[4] + self._gamma * G_fitness
            energy_returns.append(G_energy)
            fitness_returns.append(G_fitness)
        energy_returns.reverse()
        fitness_returns.reverse()

        losses: List[torch.Tensor] = []
        for entry, G_energy, G_fitness in zip(
            transitions, energy_returns, fitness_returns
        ):
            percept, move_idx, repro_action, _, _, _, _ = entry
            can_reproduce = any(percept[4 + i] < 0.5 for i in range(4))

            inp = torch.tensor(percept, dtype=torch.float32).unsqueeze(0)
            move_logits, repro_logit, V_move, V_repro = self.net.forward_with_all(inp)
            move_logits_masked = _mask_move_logits(move_logits, inp.detach())

            move_dist = torch.distributions.Categorical(
                probs=torch.softmax(move_logits_masked, dim=-1)
            )
            lp_m = move_dist.log_prob(torch.tensor(move_idx))
            ent_m = move_dist.entropy().mean()

            repro_dist = torch.distributions.Bernoulli(logits=repro_logit.squeeze())
            lp_r = repro_dist.log_prob(torch.tensor(float(repro_action)))
            ent_r = repro_dist.entropy().mean()

            G_energy_t = torch.tensor([[G_energy]], dtype=torch.float32)
            G_fitness_t = torch.tensor([[G_fitness]], dtype=torch.float32)
            adv_move = (G_energy_t - V_move).detach().squeeze()
            adv_repro = (G_fitness_t - V_repro).detach().squeeze()

            loss_actor_move = -(lp_m * adv_move + self.entropy_beta * ent_m)
            loss_actor_repro = (
                -(lp_r * adv_repro + self.entropy_beta * ent_r)
                if can_reproduce else torch.zeros((), dtype=torch.float32)
            )
            loss_critic = self.critic_coef * (
                F.mse_loss(V_move, G_energy_t) + F.mse_loss(V_repro, G_fitness_t)
            )
            losses.append(loss_actor_move + loss_actor_repro + loss_critic)

        loss = torch.stack([item.squeeze() for item in losses]).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.net.invalidate_numpy_cache()
