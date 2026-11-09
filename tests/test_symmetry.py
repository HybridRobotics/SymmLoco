"""The reflection of the G1 velocity task, checked against mjlab's G1 model and simulation."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from symmloco.tasks.velocity.mdp import (
  action_reflection,
  joint_reflection,
  observation_reflection,
)

S = np.diag([1.0, -1.0, 1.0])


def _swap_side(name: str) -> str:
  return name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")


def test_mirrored_joint_angles_mirror_every_body():
  """Forward kinematics of mirror(q) is the reflection of the kinematics of q."""
  model = get_spec().compile()
  # Joint 0 is the floating base.
  rep = joint_reflection([model.joint(i).name for i in range(1, model.njnt)])
  perm, sign = rep.perm.numpy(), rep.sign.numpy()
  data = mujoco.MjData(model)
  rng = np.random.default_rng(0)
  low, high = model.jnt_range[1:].T
  for _ in range(5):
    q = rng.uniform(low, high)
    poses = []
    for joints in (q, q[perm] * sign):
      data.qpos[:] = 0.0
      data.qpos[2] = 1.0
      data.qpos[3] = 1.0  # identity base quaternion
      data.qpos[7:] = joints
      mujoco.mj_kinematics(model, data)
      poses.append((data.xpos.copy(), data.xmat.reshape(-1, 3, 3).copy()))
    (pos, rot), (pos_m, rot_m) = poses
    for b in range(1, model.nbody):
      partner = model.body(_swap_side(model.body(b).name)).id
      np.testing.assert_allclose(pos_m[partner], S @ pos[b], atol=2e-3)
      # A reflected frame, re-expressed as a right-handed frame, is S R S.
      np.testing.assert_allclose(rot_m[partner], S @ rot[b] @ S, atol=2e-3)


def test_mirrored_state_gives_mirrored_observations():
  """Env 1 runs the mirror image of env 0; every observation term must mirror too."""
  pytest.importorskip("mujoco_warp")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  torch.manual_seed(0)
  cfg = unitree_g1_flat_env_cfg()
  cfg.scene.num_envs = 2
  cfg.observations["actor"].enable_corruption = False
  # Per-env randomization would make the two envs differ beyond the reflection.
  for name in ("encoder_bias", "base_com", "push_robot", "foot_friction"):
    cfg.events.pop(name)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    for _ in range(30):
      env.step(0.3 * torch.randn(2, 29))

    robot = env.scene["robot"]
    command = env.command_manager.get_term("twist")
    actions = action_reflection(env)
    joints = joint_reflection(robot.joint_names)
    s = torch.tensor([1.0, -1.0, 1.0])
    d, o, env1 = robot.data, env.scene.env_origins, torch.tensor([1])
    root = torch.cat(
      [
        o[1] + s * (d.root_link_pos_w[0] - o[0]),
        d.root_link_quat_w[0] * torch.tensor([1.0, -1.0, 1.0, -1.0]),
        s * d.root_link_lin_vel_w[0],
        -s * d.root_link_ang_vel_w[0],
      ]
    )
    robot.write_root_state_to_sim(root[None], env_ids=env1)
    robot.write_joint_state_to_sim(
      joints(d.joint_pos[:1]), joints(d.joint_vel[:1]), env_ids=env1
    )
    manager = env.action_manager
    for buf in (manager._action, manager._prev_action, manager._prev_prev_action):
      buf[1] = actions(buf[0])
    for buf in (command.vel_command_b, command.vel_command_w):
      buf[1] = buf[0] * torch.tensor([1.0, -1.0, -1.0])
    command.heading_target[1] = -command.heading_target[0]
    for name in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
      getattr(command, name)[1] = getattr(command, name)[0]
    command.time_left[1] = command.time_left[0]
    env.episode_length_buf[1] = env.episode_length_buf[0]
    env.sim.forward()

    for _ in range(3):
      a = 0.3 * torch.randn(1, 29)
      obs, *_ = env.step(torch.cat([a, actions(a)]))
      for group in ("actor", "critic"):
        rep = observation_reflection(env, (group,))
        # Tolerance covers the G1 model's small left-right asymmetries.
        torch.testing.assert_close(
          obs[group][1], rep(obs[group][0]), atol=0.1, rtol=0.0
        )
  finally:
    env.close()
