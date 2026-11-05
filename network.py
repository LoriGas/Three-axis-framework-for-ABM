"""Feed-forward MLP used by neural agents."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """MLP with shared backbone and four separate heads (A2C architecture).

    Backbone: input -> [Linear -> ReLU] x num_hidden_layers
    policy_head_move  (5):  movement logits
    policy_head_repro (1):  reproduction logit
    critic_move       (1):  V_move(s) — value of state for energy objective
    critic_repro      (1):  V_repro(s) — value of state for fitness objective
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 16,
        output_size: int = 6,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.policy_head_move = nn.Linear(hidden_size, output_size - 1)   # 5 logits
        self.policy_head_repro = nn.Linear(hidden_size, 1)
        self.critic_move = nn.Linear(hidden_size, 1)
        self.critic_repro = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns concatenated [move_logits(5), repro_logit(1)] for inference."""
        h = self.backbone(x)
        return torch.cat([self.policy_head_move(h), self.policy_head_repro(h)], dim=-1)

    def forward_with_all(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (move_logits, repro_logit, V_move, V_repro) — used during A2C updates."""
        h = self.backbone(x)
        return (
            self.policy_head_move(h),
            self.policy_head_repro(h),
            self.critic_move(h),
            self.critic_repro(h),
        )

    # ------------------------------------------------------------------
    # NumPy inference  (fast batch forward without PyTorch overhead)
    # ------------------------------------------------------------------

    def _cache_numpy(self) -> None:
        self._np_backbone: List[Tuple[np.ndarray, np.ndarray]] = []
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                self._np_backbone.append((
                    m.weight.detach().cpu().numpy(),
                    m.bias.detach().cpu().numpy(),
                ))
        self._np_move_W = self.policy_head_move.weight.detach().cpu().numpy()
        self._np_move_b = self.policy_head_move.bias.detach().cpu().numpy()
        self._np_repro_W = self.policy_head_repro.weight.detach().cpu().numpy()
        self._np_repro_b = self.policy_head_repro.bias.detach().cpu().numpy()

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPy forward pass. Returns concatenated [move(5), repro(1)] logits."""
        if not hasattr(self, "_np_backbone"):
            self._cache_numpy()
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        h = x
        for W, b in self._np_backbone:
            h = h @ W.T + b
            np.maximum(h, 0, out=h)  # ReLU in-place
        move = h @ self._np_move_W.T + self._np_move_b
        repro = h @ self._np_repro_W.T + self._np_repro_b
        out = np.concatenate([move, repro], axis=-1)
        return out[0] if squeeze else out

    def invalidate_numpy_cache(self) -> None:
        for attr in ("_np_backbone", "_np_move_W", "_np_move_b", "_np_repro_W", "_np_repro_b"):
            if hasattr(self, attr):
                delattr(self, attr)

    # ------------------------------------------------------------------
    # Evolution / cloning utilities
    # ------------------------------------------------------------------

    def copy_from(
        self,
        other: "NeuralNetwork",
        mutate: bool = False,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.1,
    ) -> None:
        self.load_state_dict(other.state_dict())
        if mutate:
            self._mutate_weights_inplace(mutation_rate, mutation_strength)
        self.invalidate_numpy_cache()

    def mutate_weights(self, mutation_rate: float = 0.05, mutation_strength: float = 0.1) -> None:
        self._mutate_weights_inplace(mutation_rate, mutation_strength)
        self.invalidate_numpy_cache()

    def _mutate_weights_inplace(self, mutation_rate: float, mutation_strength: float) -> None:
        with torch.no_grad():
            for param in self.parameters():
                if param.dim() == 0:
                    continue
                mask = torch.rand_like(param) < mutation_rate
                param[mask] += torch.randn_like(param)[mask] * mutation_strength
