from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.modules import SignedPermutation
from rsl_rl.utils import resolve_callable


def _rep_cfg(rep: SignedPermutation) -> dict[str, list]:
  return {"perm": rep.perm.tolist(), "sign": rep.sign.tolist()}


class EquivariantOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner for rsl_rl's ``EquivariantMLPModel`` actor and critic.

  Fills in the reflections the models and ``EquivariantGaussianDistribution`` need, using
  the task functions named in :class:`~symmloco.rl.config.EquivariantOnPolicyRunnerCfg`.
  The actor maps the observation reflection to the action reflection; the critic gets no
  output representation, which makes it invariant.
  """

  export_onnx_on_save = True

  def __init__(
    self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ) -> None:
    observation_reflection = resolve_callable(train_cfg["observation_reflection_func"])
    action_reflection = resolve_callable(train_cfg["action_reflection_func"])
    base_env = env.unwrapped
    obs_groups = train_cfg["obs_groups"]
    action_rep = action_reflection(base_env)
    actor, critic = train_cfg["actor"], train_cfg["critic"]
    actor["equivariance_cfg"] = {
      "obs": _rep_cfg(observation_reflection(base_env, obs_groups["actor"])),
      "output": _rep_cfg(action_rep),
    }
    actor["distribution_cfg"]["perm"] = action_rep.perm.tolist()
    critic["equivariance_cfg"] = {
      "obs": _rep_cfg(observation_reflection(base_env, obs_groups["critic"])),
    }
    super().__init__(env, train_cfg, log_dir, device)
