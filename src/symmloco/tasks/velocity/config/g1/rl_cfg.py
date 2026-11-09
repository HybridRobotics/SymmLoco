"""RL configuration for the Unitree G1 velocity task, one per method of the paper."""

from dataclasses import asdict, fields

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import (
  unitree_g1_ppo_runner_cfg as mjlab_unitree_g1_ppo_runner_cfg,
)

from symmloco.rl import EquivariantOnPolicyRunnerCfg, SymmlocoPpoAlgorithmCfg


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """mjlab's PPO configuration for the G1 velocity task."""
  cfg = mjlab_unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_flat_PPO"
  cfg.wandb_project = "symmloco"
  return cfg


def unitree_g1_ppoaug_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO whose mini-batches are extended with their mirror images."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.algorithm = SymmlocoPpoAlgorithmCfg(
    **asdict(cfg.algorithm),
    symmetry_cfg={
      "data_augmentation_func": "symmloco.tasks.velocity.mdp:mirror_batch",
      "use_data_augmentation": True,
    },
  )
  cfg.experiment_name = "g1_velocity_flat_PPOaug"
  return cfg


def unitree_g1_ppoeqic_runner_cfg() -> EquivariantOnPolicyRunnerCfg:
  """PPO with an equivariant actor, equivariant exploration noise and an invariant critic."""
  base = unitree_g1_ppo_runner_cfg()
  cfg = EquivariantOnPolicyRunnerCfg(
    **{f.name: getattr(base, f.name) for f in fields(base)},
    observation_reflection_func="symmloco.tasks.velocity.mdp:observation_reflection",
    action_reflection_func="symmloco.tasks.velocity.mdp:action_reflection",
  )
  cfg.actor.class_name = "EquivariantMLPModel"
  cfg.critic.class_name = "EquivariantMLPModel"
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["class_name"] = "EquivariantGaussianDistribution"
  cfg.experiment_name = "g1_velocity_flat_PPOeqic"
  return cfg
