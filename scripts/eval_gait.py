"""Evaluate the left/right gait symmetry of trained velocity policies.

Usage:
  uv run python scripts/eval_gait.py --checkpoints \
    logs/rsl_rl/g1_velocity_flat_PPO/<run_a>/<run_a>.onnx \
    logs/rsl_rl/g1_velocity_flat_PPOaug/<run_b>/<run_b>.onnx \
    logs/rsl_rl/g1_velocity_flat_PPOeqic/<run_c>/<run_c>.onnx

The policies are the ONNX files the runners export next to their checkpoints (input
``obs``, output ``actions``, observation normalization included). PPO, PPOaug and
PPOeqic all export the same plain MLP interface (PPOeqic's equivariant layers are
folded), so they are evaluated in one env. Each file is labelled by its experiment
directory (e.g. ``g1_velocity_flat_PPOeqic``); ``LABEL=path`` sets the label
explicitly. Files with the same label (e.g. one per training seed) are summarized as
mean and standard deviation, and the methods are compared in a final table.

Every policy is rolled out deterministically (mean action, no observation noise, no
pushes, no domain randomization) on a fixed set of velocity commands, one block of
robots per command. The commands stay fixed for the whole rollout, including after a
robot falls and resets.

Reported per command:

- achieved body velocity and tracking error;
- mean swing (air) and stance (contact) time of each foot, from touchdown and
  lift-off events of the ``feet_ground_contact`` sensor;
- swing / stance asymmetry ``|L - R| / mean(L, R)``, pooled over all steps of all
  robots of that command;
- number of falls.

Reported per mirrored command pair ``c = (vx, vy, wz)`` and ``Mc = (vx, -vy, -wz)``:

- velocity mirror gap ``|v(c) - M v(Mc)|`` per component, e.g. the difference between
  the achieved left and right turn rates;
- cross swing asymmetry: left foot under ``c`` against right foot under ``Mc`` (and
  vice versa), which is zero for a perfectly mirrored gait.

"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

# (label, vx, vy, wz). Commands whose mirror image is themselves.
SYMMETRIC_COMMANDS: tuple[tuple[str, float, float, float], ...] = (
  ("fwd_0.3", 0.3, 0.0, 0.0),
  ("fwd_0.6", 0.6, 0.0, 0.0),
  ("fwd_1.0", 1.0, 0.0, 0.0),
  ("back_0.5", -0.5, 0.0, 0.0),
)
# (label, vx, vy, wz). The mirror command (vx, -vy, -wz) is evaluated as well.
MIRRORED_COMMANDS: tuple[tuple[str, float, float, float], ...] = (
  ("lateral_0.5", 0.0, 0.5, 0.0),
  ("fwd_0.5_yaw_0.3", 0.5, 0.0, 0.3),
)
_MIRROR = (1.0, -1.0, -1.0)
_EPS = 1e-9


@dataclass(frozen=True)
class EvalGaitConfig:
  checkpoints: tuple[str, ...] = ()
  """Exported ONNX policies, ``path`` or ``LABEL=path``. Without a label, the
  experiment directory name (two levels up) is used."""
  task: str = "VelocityFlat-G1-PPO"
  """Task whose play env is used. The PPO, PPOaug and PPOeqic tasks share one env."""
  envs_per_command: int = 64
  warmup_s: float = 2.0
  """Seconds after each (re)set that are excluded from the statistics."""
  min_phase_s: float = 0.05
  """Swing / stance phases shorter than this are contact flicker, not steps, and are
  ignored."""
  duration_s: float = 12.0
  """Rollout length per checkpoint, warmup included."""
  device: str | None = None
  output: str | None = None
  """Optional JSON file for the per-checkpoint results and the summary."""


def _commands() -> tuple[list[str], torch.Tensor, list[tuple[str, int, int]]]:
  labels: list[str] = []
  values: list[tuple[float, float, float]] = []
  for label, vx, vy, wz in SYMMETRIC_COMMANDS:
    labels.append(label)
    values.append((vx, vy, wz))
  pairs: list[tuple[str, int, int]] = []
  for label, vx, vy, wz in MIRRORED_COMMANDS:
    pairs.append((label, len(labels), len(labels) + 1))
    labels += [f"{label}+", f"{label}-"]
    values += [(vx, vy, wz), (vx * _MIRROR[0], vy * _MIRROR[1], wz * _MIRROR[2])]
  return labels, torch.tensor(values), pairs


def _asym(a: float, b: float) -> float:
  mean = 0.5 * (a + b)
  return abs(a - b) / mean if mean > _EPS else math.nan


def _make_env(task_id: str, cfg: EvalGaitConfig, num_envs: int, device: str):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # No domain randomization: drop every startup (model randomization) and interval
  # (push) event; reset events only place the robot.
  removed = [
    k for k, ev in env_cfg.events.items() if ev.mode in ("startup", "interval")
  ]
  for k in removed:
    del env_cfg.events[k]
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.observations["critic"].enable_corruption = False
  env_cfg.curriculum = {}
  print(f"[INFO] Removed randomization events: {removed}")
  print(f"[INFO] Remaining events: {list(env_cfg.events)}")

  twist = env_cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.debug_vis = False
  r = twist.ranges
  for label, vx, vy, wz in SYMMETRIC_COMMANDS + MIRRORED_COMMANDS:
    inside = (
      r.lin_vel_x[0] <= vx <= r.lin_vel_x[1]
      and r.lin_vel_y[0] <= abs(vy) <= r.lin_vel_y[1]
      and r.ang_vel_z[0] <= abs(wz) <= r.ang_vel_z[1]
    )
    if not inside:
      print(f"[WARN] Command {label} lies outside the training range {r}.")

  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg


@dataclass
class _Stats:
  """Per-command accumulators (index: command, foot)."""

  n_cmd: int
  device: str
  swing_sum: torch.Tensor = field(init=False)
  swing_cnt: torch.Tensor = field(init=False)
  stance_sum: torch.Tensor = field(init=False)
  stance_cnt: torch.Tensor = field(init=False)
  vel_sum: torch.Tensor = field(init=False)
  vel_cnt: torch.Tensor = field(init=False)
  falls: torch.Tensor = field(init=False)

  def __post_init__(self):
    z = lambda *s: torch.zeros(*s, device=self.device, dtype=torch.float64)  # noqa: E731
    self.swing_sum, self.swing_cnt = z(self.n_cmd, 2), z(self.n_cmd, 2)
    self.stance_sum, self.stance_cnt = z(self.n_cmd, 2), z(self.n_cmd, 2)
    self.vel_sum, self.vel_cnt = z(self.n_cmd, 3), z(self.n_cmd)
    self.falls = z(self.n_cmd)


def _rollout(env, policy, cmd_idx, cmd_values, foot_idx, cfg, n_cmd) -> _Stats:
  base = env.unwrapped
  dt = base.step_dt
  device = base.device
  robot = base.scene["robot"]
  sensor = base.scene["feet_ground_contact"]
  stats = _Stats(n_cmd, device)

  def per_cmd(values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((n_cmd,) + values.shape[1:], device=device, dtype=torch.float64)
    return out.index_add_(0, cmd_idx, values.to(torch.float64))

  obs, _ = env.reset()
  t_alive = torch.zeros(base.num_envs, device=device)
  for _ in range(round(cfg.duration_s / dt)):
    # no_grad on the policy only: inference_mode around env.step would turn the env's
    # buffers into inference tensors, which a later reset cannot write to.
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    t_alive += dt
    done = dones.bool()
    t_alive[done] = 0.0
    stats.falls += per_cmd((base.reset_terminated & done).float())

    valid = (t_alive > cfg.warmup_s).unsqueeze(-1)
    data = sensor.data
    touchdown = sensor.compute_first_contact(dt) & valid
    liftoff = sensor.compute_first_air(dt) & valid
    swing = data.last_air_time[:, foot_idx]
    stance = data.last_contact_time[:, foot_idx]
    touchdown, liftoff = touchdown[:, foot_idx], liftoff[:, foot_idx]
    touchdown &= swing >= cfg.min_phase_s
    liftoff &= stance >= cfg.min_phase_s
    stats.swing_sum += per_cmd(torch.where(touchdown, swing, 0.0))
    stats.swing_cnt += per_cmd(touchdown.float())
    stats.stance_sum += per_cmd(torch.where(liftoff, stance, 0.0))
    stats.stance_cnt += per_cmd(liftoff.float())

    vel = torch.cat(
      [robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]],
      -1,
    )
    v = valid.squeeze(-1)
    stats.vel_sum += per_cmd(torch.where(v.unsqueeze(-1), vel, 0.0))
    stats.vel_cnt += per_cmd(v.float())
  del cmd_values
  return stats


def _summarize(stats: _Stats, labels, cmd_values, pairs) -> dict:
  nan = torch.tensor(math.nan, dtype=torch.float64, device=stats.swing_sum.device)
  swing = torch.where(stats.swing_cnt > 0, stats.swing_sum / stats.swing_cnt, nan).cpu()
  stance = torch.where(
    stats.stance_cnt > 0, stats.stance_sum / stats.stance_cnt, nan
  ).cpu()
  vel = (stats.vel_sum / stats.vel_cnt.clamp(min=1).unsqueeze(-1)).cpu()
  steps = stats.swing_cnt.cpu()
  falls = stats.falls.cpu()

  commands = {}
  for i, label in enumerate(labels):
    c = cmd_values[i].double()
    commands[label] = {
      "command": c.tolist(),
      "achieved": vel[i].tolist(),
      "lin_vel_error": float((vel[i, :2] - c[:2]).norm()),
      "ang_vel_error": float((vel[i, 2] - c[2]).abs()),
      "swing_L": float(swing[i, 0]),
      "swing_R": float(swing[i, 1]),
      "stance_L": float(stance[i, 0]),
      "stance_R": float(stance[i, 1]),
      "swing_asym": _asym(float(swing[i, 0]), float(swing[i, 1])),
      "stance_asym": _asym(float(stance[i, 0]), float(stance[i, 1])),
      "steps_L": int(steps[i, 0]),
      "steps_R": int(steps[i, 1]),
      "falls": int(falls[i]),
    }

  mirror = torch.tensor(_MIRROR, dtype=torch.float64)
  pair_out = {}
  for label, ip, im in pairs:
    gap = (vel[ip] - mirror * vel[im]).abs()
    cross = 0.5 * (
      _asym(float(swing[ip, 0]), float(swing[im, 1]))
      + _asym(float(swing[ip, 1]), float(swing[im, 0]))
    )
    pair_out[label] = {
      "gap_vx": float(gap[0]),
      "gap_vy": float(gap[1]),
      "gap_wz": float(gap[2]),
      "cross_swing_asym": cross,
    }

  sym = [s[0] for s in SYMMETRIC_COMMANDS]
  overall = {
    "mean_swing_asym_symmetric": _nanmean([commands[k]["swing_asym"] for k in sym]),
    "mean_stance_asym_symmetric": _nanmean([commands[k]["stance_asym"] for k in sym]),
    "mean_cross_swing_asym_pairs": _nanmean(
      [p["cross_swing_asym"] for p in pair_out.values()]
    ),
    "mean_lin_vel_error": _nanmean([c["lin_vel_error"] for c in commands.values()]),
    "mean_ang_vel_error": _nanmean([c["ang_vel_error"] for c in commands.values()]),
    "total_falls": sum(c["falls"] for c in commands.values()),
  }
  return {"commands": commands, "pairs": pair_out, "overall": overall}


def _nanmean(xs: list[float]) -> float:
  xs = [x for x in xs if not math.isnan(x)]
  return sum(xs) / len(xs) if xs else math.nan


def _print_result(name: str, res: dict) -> None:
  print(f"\n=== {name} ===")
  print(
    f"{'command':<18}{'achieved (vx vy wz)':>24}{'swing L/R [s]':>16}"
    f"{'swing asym':>12}{'stance asym':>13}{'steps':>8}{'falls':>7}"
  )
  for label, c in res["commands"].items():
    a = " ".join(f"{x:+.2f}" for x in c["achieved"])
    print(
      f"{label:<18}{a:>24}{c['swing_L']:>8.3f}/{c['swing_R']:<7.3f}"
      f"{100 * c['swing_asym']:>11.1f}%{100 * c['stance_asym']:>12.1f}%"
      f"{c['steps_L'] + c['steps_R']:>8}{c['falls']:>7}"
    )
  print(
    f"\n{'mirrored pair':<18}{'gap vx':>9}{'gap vy':>9}{'gap wz':>9}{'cross swing asym':>18}"
  )
  for label, p in res["pairs"].items():
    print(
      f"{label:<18}{p['gap_vx']:>9.3f}{p['gap_vy']:>9.3f}{p['gap_wz']:>9.3f}"
      f"{100 * p['cross_swing_asym']:>17.1f}%"
    )


def _aggregate(results: list[dict]) -> dict[str, dict[str, float]]:
  out = {}
  for k in results[0]["overall"]:
    xs = torch.tensor([float(r["overall"][k]) for r in results], dtype=torch.float64)
    xs = xs[~xs.isnan()]
    out[k] = {
      "mean": float(xs.mean()) if len(xs) else math.nan,
      "std": float(xs.std(unbiased=False)) if len(xs) else math.nan,
      "n": len(xs),
    }
  return out


def _parse_checkpoint(spec: str) -> tuple[str, Path]:
  label, sep, path = spec.partition("=")
  if not sep:
    path = spec
    label = Path(spec).resolve().parent.parent.name
  return label, Path(path)


class _OnnxPolicy:
  """Batched ONNX policy: obs TensorDict -> actions on the env device."""

  def __init__(self, path: Path, env) -> None:
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    self.metadata = {p.key: p.value for p in model.metadata_props}
    self._check(path, model, env)
    # The export has a fixed batch size of 1; the graph (normalizer + MLP) is
    # batch-independent, so the batch dimension is made dynamic.
    for v in list(model.graph.input) + list(model.graph.output):
      v.type.tensor_type.shape.dim[0].dim_param = "batch"
    providers = [
      p
      for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
      if p in ort.get_available_providers()
    ]
    self._session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    self._input = model.graph.input[0].name
    self._device = env.unwrapped.device

  def _check(self, path: Path, model, env) -> None:
    base = env.unwrapped
    robot = base.scene["robot"]
    expected = {
      "joint_names": list(robot.joint_names),
      "observation_names": list(base.observation_manager.active_terms["actor"]),
    }
    for key, value in expected.items():
      if key in self.metadata and self.metadata[key].split(",") != value:
        raise ValueError(f"{path}: {key} does not match the eval env.")
    obs_dim = base.observation_manager.group_obs_dim["actor"][0]
    act_dim = base.action_manager.total_action_dim
    in_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    out_dim = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    if (in_dim, out_dim) != (obs_dim, act_dim):
      raise ValueError(
        f"{path}: policy maps {in_dim} -> {out_dim}, env needs {obs_dim} -> {act_dim}."
      )
    if "action_scale" in self.metadata:
      scale = base.action_manager.get_term("joint_pos")._scale
      scale = scale[0].cpu().numpy() if isinstance(scale, torch.Tensor) else scale
      onnx_scale = np.array(
        [float(x) for x in self.metadata["action_scale"].split(",")]
      )
      if not np.allclose(onnx_scale, scale, atol=1e-3):
        raise ValueError(f"{path}: action_scale does not match the eval env.")

  def __call__(self, obs) -> torch.Tensor:
    x = obs["actor"].detach().cpu().numpy().astype(np.float32)
    y = self._session.run(None, {self._input: x})[0]
    return torch.from_numpy(y).to(self._device)


def _print_comparison(summary: dict[str, dict]) -> None:
  rows = [
    ("swing asym (symmetric cmds)", "mean_swing_asym_symmetric", 100, "%"),
    ("stance asym (symmetric cmds)", "mean_stance_asym_symmetric", 100, "%"),
    ("cross swing asym (pairs)", "mean_cross_swing_asym_pairs", 100, "%"),
    ("lin vel error [m/s]", "mean_lin_vel_error", 1, ""),
    ("ang vel error [rad/s]", "mean_ang_vel_error", 1, ""),
    ("falls", "total_falls", 1, ""),
  ]
  labels = list(summary)
  width = max(18, *(len(x) + 2 for x in labels))
  print("\n=== Comparison (mean ± std over checkpoints) ===")
  print(f"{'':<30}" + "".join(f"{x:>{width}}" for x in labels))
  print(f"{'checkpoints':<30}" + "".join(f"{summary[x]['n']:>{width}}" for x in labels))
  for name, key, scale, unit in rows:
    cells = []
    for x in labels:
      s = summary[x]["metrics"][key]
      cells.append(f"{scale * s['mean']:.3g} ± {scale * s['std']:.2g}{unit}")
    print(f"{name:<30}" + "".join(f"{c:>{width}}" for c in cells))


def run_eval(cfg: EvalGaitConfig) -> dict:
  from mjlab.utils.torch import configure_torch_backends

  if not cfg.checkpoints:
    raise ValueError("Pass --checkpoints with one or more .onnx files.")
  ckpts = [_parse_checkpoint(c) for c in cfg.checkpoints]
  for _, path in ckpts:
    if not path.exists():
      raise FileNotFoundError(path)
    if path.suffix != ".onnx":
      raise ValueError(f"{path}: expected an exported .onnx policy.")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  labels, cmd_values, pairs = _commands()
  n_cmd = len(labels)
  env, _ = _make_env(cfg.task, cfg, n_cmd * cfg.envs_per_command, device)
  base = env.unwrapped

  # Fixed command per env: block i of envs_per_command robots gets command i, also
  # after a reset (the term's resampling is replaced).
  cmd_idx = torch.arange(n_cmd, device=device).repeat_interleave(cfg.envs_per_command)
  fixed = cmd_values.to(device)[cmd_idx]
  term = base.command_manager.get_term("twist")

  def _fixed_resample(env_ids: torch.Tensor) -> None:
    term.vel_command_b[env_ids] = fixed[env_ids]
    term.vel_command_w[env_ids] = fixed[env_ids]

  term._resample_command = _fixed_resample

  sensor = base.scene["feet_ground_contact"]
  names = sensor.primary_names
  foot_idx = torch.tensor(
    [next(i for i, n in enumerate(names) if side in n) for side in ("left", "right")],
    device=device,
  )
  print(f"[INFO] Feet (left, right): {[names[i] for i in foot_idx.tolist()]}")

  results: dict[str, dict] = {}
  for label, path in ckpts:
    policy = _OnnxPolicy(path, env)
    stats = _rollout(env, policy, cmd_idx, cmd_values, foot_idx, cfg, n_cmd)
    res = _summarize(stats, labels, cmd_values, pairs)
    res["label"] = label
    res["path"] = str(path)
    res["run"] = policy.metadata.get("run_path", "")
    key = f"{label}: {path}"
    if key in results:
      key = f"{key} [{len(results)}]"
    results[key] = res
    _print_result(key, res)

  summary: dict[str, dict] = {}
  for label in dict.fromkeys(r["label"] for r in results.values()):
    group = [r for r in results.values() if r["label"] == label]
    summary[label] = {"n": len(group), "metrics": _aggregate(group)}
  _print_comparison(summary)

  out = {
    "config": asdict(cfg),
    "checkpoints": results,
    "summary": summary,
  }
  if cfg.output:
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2))
    print(f"[INFO] Wrote {cfg.output}")
  env.close()
  return out


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  import symmloco.tasks  # noqa: F401

  # Space-separated collections: --checkpoints a.onnx b.onnx
  cfg = tyro.cli(
    EvalGaitConfig,
    prog=sys.argv[0],
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_eval(cfg)


if __name__ == "__main__":
  main()
