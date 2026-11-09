"""Left-right reflection of the velocity task as signed permutations.

The reflection mirrors the world through the xz-plane (y -> -y) and swaps the robot's left
and right sides. Every observation term and the action vector then transform as a signed
permutation ``x -> sign * x[perm]``, which rsl_rl uses in two ways: ``mirror_batch`` is
the data augmentation function of its ``Symmetry`` extension (PPOaug), and
``observation_reflection`` and ``action_reflection`` parameterize ``EquivariantMLPModel``
through :class:`~symmloco.rl.EquivariantOnPolicyRunner` (PPOeqic).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from rsl_rl.modules import SignedPermutation
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "TERM_KINDS",
  "action_reflection",
  "joint_reflection",
  "mirror_batch",
  "observation_reflection",
]

TERM_KINDS: dict[str, str] = {
  "base_lin_vel": "vector",
  "base_ang_vel": "pseudovector",
  "projected_gravity": "vector",
  "joint_pos": "joint",
  "joint_vel": "joint",
  "actions": "action",
  "command": "command",
  "foot_height": "feet",
  "foot_air_time": "feet",
  "foot_contact": "feet",
  "foot_contact_forces": "feet_vector",
}
"""How each observation term of the velocity task transforms, by term name.

vector: xyz vector in the body or world frame, y flips. pseudovector: angular velocity, x
and z flip. joint / action: values in robot joint order / action order (see
:func:`joint_reflection`). command: body-frame (vx, vy, wz), vy and wz flip. feet: one
value per foot, feet swap. feet_vector: one world-frame xyz vector per foot, feet swap
and y flips.
"""


def mirror_batch(
  env, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append the mirrored samples to a batch (rsl_rl ``data_augmentation_func``)."""
  base_env = env.unwrapped
  if obs is not None:
    mirrored = {k: observation_reflection(base_env, (k,))(obs[k]) for k in obs.keys()}
    obs = torch.cat([obs, TensorDict(mirrored, batch_size=obs.batch_size)], dim=0)
  if actions is not None:
    actions = torch.cat([actions, action_reflection(base_env)(actions)], dim=0)
  return obs, actions


def observation_reflection(
  env: ManagerBasedRlEnv, groups: Sequence[str]
) -> SignedPermutation:
  """Reflection of the concatenation of the given observation groups."""
  manager = env.observation_manager
  joints = joint_reflection(env.scene["robot"].joint_names)
  actions = action_reflection(env)
  reps = []
  for group in groups:
    names = manager.active_terms[group]
    for name, dims in zip(names, manager.group_obs_term_dim[group], strict=True):
      if manager.get_term_cfg(group, name).history_length > 0:
        raise NotImplementedError(f"observation term '{group}/{name}' has a history")
      if name not in TERM_KINDS:
        raise ValueError(
          f"observation term '{group}/{name}' has no entry in TERM_KINDS"
        )
      reps.append(_term_reflection(TERM_KINDS[name], math.prod(dims), joints, actions))
  return _concat(reps)


def action_reflection(env: ManagerBasedRlEnv) -> SignedPermutation:
  """Reflection of the action vector, from the joints each action term targets."""
  manager = env.action_manager
  return _concat(
    [
      joint_reflection(manager.get_term(name).target_names)
      for name in manager.active_terms
    ]
  )


def joint_reflection(joint_names: Sequence[str]) -> SignedPermutation:
  """Reflection of joint-indexed values, from the joint names.

  ``left_*`` and ``right_*`` joints swap. A joint rotating about its body's x or z axis
  (roll, yaw) changes sign, one rotating about y (pitch, knees, elbows) does not; this
  holds for robots whose left and right chains share their joint axes, as the G1 does.
  """
  index = {name: i for i, name in enumerate(joint_names)}

  def partner(name: str) -> str:
    if name.startswith("left_"):
      return "right_" + name.removeprefix("left_")
    if name.startswith("right_"):
      return "left_" + name.removeprefix("right_")
    return name

  perm = [index[partner(name)] for name in joint_names]
  sign = [-1.0 if ("_roll_" in n or "_yaw_" in n) else 1.0 for n in joint_names]
  return SignedPermutation(perm, sign)


_SIGN_PATTERNS = {
  "vector": (1.0, -1.0, 1.0),
  "pseudovector": (-1.0, 1.0, -1.0),
  "command": (1.0, -1.0, -1.0),
}


def _term_reflection(
  kind: str, dim: int, joints: SignedPermutation, actions: SignedPermutation
) -> SignedPermutation:
  if kind in ("joint", "action"):
    rep = joints if kind == "joint" else actions
    if len(rep) != dim:
      raise ValueError(f"{kind} term has dimension {dim}, expected {len(rep)}")
    return rep
  if kind in ("feet", "feet_vector"):
    half = dim // 2
    perm = torch.cat([torch.arange(half, dim), torch.arange(half)])
    pattern = (1.0, -1.0, 1.0) if kind == "feet_vector" else (1.0,)
    return SignedPermutation(perm, torch.tensor(pattern).repeat(dim // len(pattern)))
  pattern = _SIGN_PATTERNS[kind]
  if dim % len(pattern) != 0:
    raise ValueError(f"{kind} term has dimension {dim}")
  sign = torch.tensor(pattern).repeat(dim // len(pattern))
  return SignedPermutation(torch.arange(dim), sign)


def _concat(reps: Sequence[SignedPermutation]) -> SignedPermutation:
  perms, offset = [], 0
  for rep in reps:
    perms.append(rep.perm + offset)
    offset += len(rep)
  return SignedPermutation(torch.cat(perms), torch.cat([rep.sign for rep in reps]))
