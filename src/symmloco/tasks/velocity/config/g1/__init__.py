from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from symmloco.rl import EquivariantOnPolicyRunner

from .rl_cfg import (
  unitree_g1_ppo_runner_cfg,
  unitree_g1_ppoaug_runner_cfg,
  unitree_g1_ppoeqic_runner_cfg,
)


def g1_flat_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50

  # Fixed command ranges (mjlab's first curriculum stage), no velocity curriculum.
  cfg.curriculum.pop("command_vel", None)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  return cfg


register_mjlab_task(
  task_id="VelocityFlat-G1-PPO",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="VelocityFlat-G1-PPOaug",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppoaug_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="VelocityFlat-G1-PPOeqic",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppoeqic_runner_cfg(),
  runner_cls=EquivariantOnPolicyRunner,
)
