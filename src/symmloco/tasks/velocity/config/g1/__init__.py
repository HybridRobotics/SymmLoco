from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from symmloco.rl import EquivariantOnPolicyRunner

from .rl_cfg import (
  unitree_g1_ppo_runner_cfg,
  unitree_g1_ppoaug_runner_cfg,
  unitree_g1_ppoeqic_runner_cfg,
)

register_mjlab_task(
  task_id="VelocityFlat-G1-PPO",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="VelocityFlat-G1-PPOaug",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppoaug_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="VelocityFlat-G1-PPOeqic",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppoeqic_runner_cfg(),
  runner_cls=EquivariantOnPolicyRunner,
)
