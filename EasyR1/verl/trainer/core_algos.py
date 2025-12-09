# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    EMA_GRPO = "ema_grpo" 


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns






class _EmaStdTracker:
    """
    Maintain EMA first moment E[X] and second moment E[X^2] for each "task key",
    and compute std = sqrt(E[X^2] - (E[X])^2).
    - Here X refers to the "Outcome score" (a single scalar obtained by summing token-level rewards over time).
    - The statistics are persistent only within a single process / rank; in multi-process settings,
      each process maintains its own tracker (for global consistency, use all-reduce or save to trainer state externally).
    - Benefit of EMA: adaptive to non-stationary distributions; meanwhile avoids over-reliance on early statistics
      (the effective memory length is controlled by decay).
    """
    def __init__(self, decay: float = 0.99, min_std: float = 1e-3):
        self.decay = decay          # EMA decay factor; larger means longer memory
        self.min_std = min_std      # Lower bound of std for numerical stability to avoid division by zero
        # Mapping: task_key -> { "m1": E[X], "m2": E[X^2], "initialized": bool }
        self.state: dict[str, dict[str, float]] = {}

    def get_std(self, key: str) -> float:
        """
        Get the current EMA standard deviation of the given task.
        If uninitialized, return min_std (a conservative value to avoid explosion / NaN at the beginning).
        """
        s = self.state.get(key)
        if not s or not s.get("initialized", False):
            return self.min_std
        # Numerical stability: E[X^2] - (E[X])^2 can be slightly negative due to floating-point errors; clamp it
        var = max(s["m2"] - s["m1"] * s["m1"], 0.0)
        # Also clamp std with a lower bound to prevent scaling explosion caused by extremely small variance
        return float(max(var, self.min_std ** 2) ** 0.5)

    def update_with_batch_scores(self, key: str, scores: torch.Tensor):
        """
        Update EMA first and second moments using the "current batch of outcome scores" for this task.
        Convention: normalization uses the *old EMA values*, and EMA is updated *afterward* using this batch,
        to avoid bias caused by "seeing itself".
        """
        if scores.numel() == 0:
            return
        x = scores.float()
        # Directly use mean and second moment for EMA (not unbiased variance;
        # EMA itself is a biased estimator but is more robust for online / non-stationary settings)
        m1_batch = x.mean().item()
        m2_batch = (x * x).mean().item()

        s = self.state.get(key)
        if s is None or not s.get("initialized", False):
            # First time seeing this task: initialize with current batch statistics
            self.state[key] = {"m1": m1_batch, "m2": m2_batch, "initialized": True}
        else:
            d = self.decay
            s["m1"] = d * s["m1"] + (1.0 - d) * m1_batch
            s["m2"] = d * s["m2"] + (1.0 - d) * m2_batch
            s["initialized"] = True

    # ===== NEW: get the current EMA mean for this task (return 0.0 if uninitialized) =====
    def get_mean(self, key: str) -> float:
        """
        Return the current EMA mean (E[X]) of the task. Return 0.0 if uninitialized.
        (Used only for logging / monitoring; not involved in algorithmic scaling; no clipping applied.)
        """
        s = self.state.get(key)
        if not s or not s.get("initialized", False):
            return 0.0
        return float(s["m1"])

    # ===== NEW: print means and stds in two lines =====
    def log_means_and_stds(
        self,
        keys: list[str] | None = None,
        print_fn=None,
        tag_means: str = "EMA_MEAN",
        tag_stds: str = "EMA_STD",
        digits: int = 6,
        sort_keys: bool = True,
    ) -> None:
        """
        Print in two lines:
          [EMA_MEAN] task=mean:...
          [EMA_STD]  task=std:...
        """
        if print_fn is None:
            print_fn = print
        if keys is None:
            keys = list(self.state.keys())
        if sort_keys:
            keys = sorted(keys)
        if not keys:
            try:
                print_fn(f"[{tag_means}] (empty)")
                print_fn(f"[{tag_stds}] (empty)")
            except Exception:
                pass
            return

        mean_line = "[{tag}] ".format(tag=tag_means) + ", ".join(
            f"{k}=mean:{self.get_mean(k):.{digits}f}" for k in keys
        )
        std_line = "[{tag}] ".format(tag=tag_stds) + ", ".join(
            f"{k}=std:{self.get_std(k):.{digits}f}" for k in keys
        )
        try:
            print_fn(mean_line)
            print_fn(std_line)
        except Exception:
            # Defensive: avoid logger / print_fn failure affecting training
            pass





# Global (process-local) singleton; for cross-process sharing, synchronize outside
_EMA_STD_TRACKER = _EmaStdTracker()


# ===== NEW: task key generation function =====
def _task_key_of(sample_problem_type: str, sample_data_type: str | None) -> str:
    """
    Task partition rule:
    - By default, aggregate by problem_type;
    - If problem_type == "segmentation", further split by data_type into
      "segmentation/image" and "segmentation/video".
    """
    if sample_problem_type == "segmentation":
        dt = (sample_data_type or "").lower()
        if dt in ("video", "image"):
            return f"segmentation/{dt}"
    return sample_problem_type



@register_adv_estimator(AdvantageEstimator.EMA_GRPO)
def compute_ema_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,        # (bs, response_length)
    response_mask: torch.Tensor,              # (bs, response_length)
    index,                                    # (bs,) group id: numpy array / list[str] / torch.Tensor (same prompt shares)
    problem_type,                             # (bs,) numpy/list/torch.Tensor: strings
    data_type=None,                           # (bs,) numpy/list/torch.Tensor/None: strings; only used to split segmentation image/video
    # ===== Tunable hyperparameters (can be passed through AlgorithmConfig) =====
    ema_decay: float = 0.99,                  # EMA decay; can be smaller (e.g., 0.97) for non-stationary tasks
    min_std: float = 1e-3,                    # Lower bound for numerical stability
    use_group_mean_centering: bool = True,    # Whether to keep GRPO-style group-wise mean-centering (recommended True)
    eps: float = 1e-6,                        # Division-by-zero protection during scaling
    # NEW: guard rail: fallback to group std if the threshold is exceeded (default ±5)
    guard_abs_max: float = 5.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EMA-GRPO (Outcome supervision):
    1) Keep GRPO-style group-wise mean-centering (samples with the same index are centered),
       to reduce intra-group variance;
    2) No longer use "group std" for scaling; instead use "task-level EMA std";
    3) **First update EMA using current batch scores (including initialization),
       then scale using the "updated" std**,
       i.e., statistics that have "seen themselves", improving responsiveness to non-stationary distributions;
    4) **Guard rail**: if after scaling with "task-level std", any value in a group
       (same index) exceeds [-guard_abs_max, +guard_abs_max],
       then that group falls back to "group std" scaling (only that group is affected).
    Returns:
        advantages: (bs, response_length)
        returns:    (bs, response_length)
    """
    # —— Device / dtype alignment (consistent with original file conventions) ——
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    # —— Set global tracker hyperparameters (if desired to set only once, one can add a flag;
    # here we directly assign for clarity) ——
    _EMA_STD_TRACKER.decay = float(ema_decay)
    _EMA_STD_TRACKER.min_std = float(min_std)

    # —— Convert index / problem_type / data_type to Python lists uniformly
    # (robustly handling numpy / torch / list) ——
    def _to_list(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.tolist()
        else:
            try:
                import numpy as _np
                if isinstance(x, _np.ndarray):
                    x = x.tolist()
            except Exception:
                pass
        if not isinstance(x, (list, tuple)):
            x = list(x)
        return x

    index_list = _to_list(index)
    problem_type_list = _to_list(problem_type)
    data_type_list = _to_list(data_type)

    # Note: index may be UUID or any hashable object; convert to str uniformly as the grouping key
    index_keys = [str(g) for g in index_list]

    # —— Aggregate token-level reward into a single "Outcome score":
    # in most scenarios, only the last token is outcome and others are 0 ——
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    bsz = scores.shape[0]

    # —— Group sample positions by "task key" ——
    # Task partition rule: default by problem_type;
    # if problem_type == "segmentation", further split by data_type
    def _task_key_of(sample_problem_type: str, sample_data_type: str | None) -> str:
        if sample_problem_type == "segmentation":
            dt = (sample_data_type or "").lower()
            if dt in ("video", "image"):
                return f"segmentation/{dt}"
        return sample_problem_type

    task_to_pos: dict[str, list[int]] = {}
    for i in range(bsz):
        pt = str(problem_type_list[i]) if problem_type_list is not None else ""
        dt = None
        if pt == "segmentation" and data_type_list is not None:
            dt = str(data_type_list[i]) if i < len(data_type_list) else None
        key = _task_key_of(pt, dt)
        task_to_pos.setdefault(key, []).append(i)

    # —— Group-wise mean-centering (multiple samples with the same index) ——
    centered = scores.clone()
    gid_to_pos: dict[str, list[int]] = {}
    for i, gid in enumerate(index_keys):
        gid_to_pos.setdefault(gid, []).append(i)

    if use_group_mean_centering:
        for gid, pos_list in gid_to_pos.items():
            # GRPO / EMA-GRPO assumption: each group must have at least 2 samples
            assert len(pos_list) > 1, "EMA-GRPO requires rollout.n > 1 per group (same index)."
            g = scores[pos_list]
            g_mean = g.mean()
            centered[pos_list] = g - g_mean
    else:
        # If centering is disabled, this is a no-op; keep shape consistency
        centered = centered - 0.0

    # =========================
    # First use "current batch scores" to **update / initialize**
    # the EMA statistics for each task,
    # then read the "updated" task std for scaling.
    # =========================

    # First update / initialize EMA for each task using this batch's scores
    for key, pos_list in task_to_pos.items():
        _EMA_STD_TRACKER.update_with_batch_scores(key, scores[pos_list])

    # 【Logging】current task stds (statistics already include this batch)
    _EMA_STD_TRACKER.log_means_and_stds(
        keys=None,
        tag_means="EMA_MEAN(after_update)",
        tag_stds="EMA_STD(after_update)",
    )


    # Then read the "updated" std for scaling (statistics that have "seen themselves")
    scaled = centered.clone()

    # —— For each "task key", further split into "groups" (same index) for guard-rail checking ——
    from collections import defaultdict as _dd
    for key, task_pos in task_to_pos.items():
        task_std = _EMA_STD_TRACKER.get_std(key)  # already includes contribution from this batch

        # Under this task, further group samples by "group id (index_keys)"
        group_to_pos: dict[str, list[int]] = _dd(list)
        for i in task_pos:
            group_to_pos[index_keys[i]].append(i)

        # For each group, first try scaling with "task-level std";
        # if any value exceeds the guard, fall back to "group std"
        for gid, gpos in group_to_pos.items():
            # 1) First scale this group with the task-level std
            tmp = centered[gpos] / (task_std + eps)

            # 2) Guard check: if any value exceeds [-guard_abs_max, +guard_abs_max], fall back
            if torch.any(torch.abs(tmp) > guard_abs_max):
                # Group std: follow original GRPO logic, use the group-wise std of scores
                g_scores = scores[gpos].float()
                g_std = torch.std(g_scores, unbiased=False).item()
                scaled[gpos] = centered[gpos] / (g_std + eps)
            else:
                # No violation: keep task-level std scaling
                scaled[gpos] = tmp

    # —— Broadcast back to token dimension and multiply by mask
    # (same return convention as the original implementation: for pure outcome,
    #  advantages == returns) ——
    if response_mask.device != device:
        response_mask = response_mask.to(device)
    if not torch.is_floating_point(response_mask):
        response_mask = response_mask.to(dtype)

    returns = scaled.to(dtype).unsqueeze(-1) * response_mask  # (bs, T)
    advantages = returns  # Pure outcome supervision: A = R

    return advantages, returns





@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp negative_approx_kl to avoid nan kld
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    pg_loss = -advantages * ratio  # -ratio * A
    pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
    pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
