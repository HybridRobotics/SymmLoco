"""RL configuration classes beyond mjlab's."""

from dataclasses import dataclass
from typing import Any

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class SymmlocoPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO configuration plus rsl_rl's symmetry extension."""

  symmetry_cfg: dict[str, Any] | None = None
  """Keyword arguments of ``rsl_rl.extensions.Symmetry``; ``None`` disables it."""


@dataclass(kw_only=True)
class EquivariantOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration for rsl_rl's ``EquivariantMLPModel`` actor and critic.

  The task supplies the reflections as ``module:function`` paths, resolved like rsl_rl's
  ``data_augmentation_func``.
  """

  observation_reflection_func: str
  """Called as ``f(env, groups)``; returns the ``SignedPermutation`` of the concatenated
  observation groups."""

  action_reflection_func: str
  """Called as ``f(env)``; returns the ``SignedPermutation`` of the action vector."""
