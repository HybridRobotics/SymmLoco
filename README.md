# SymmLoco on mjlab

The two symmetry methods of **SymmLoco** ("Leveraging Symmetry in RL-based Legged
Locomotion Control", IROS 2024), PPO with symmetry data augmentation (PPOaug) and PPO with
an equivariant actor-critic (PPOeqic), are now implemented in rsl_rl: the augmentation in
its `Symmetry` extension, the equivariant networks on its
[`extras`](https://github.com/leggedrobotics/rsl_rl/tree/extras) branch. To make them easy
to use in [mjlab](https://github.com/mujocolab/mjlab), this repository is an mjlab example:
it trains mjlab's own Unitree G1 flat-terrain velocity task with PPO, PPOaug and PPOeqic.
The original Isaac Gym implementation lives on the `isaacgym/main` and
`isaacgym/dribbling` branches.

| id suffix | method | implementation |
|---|---|---|
| `-PPO` | PPO baseline | rsl_rl `MLPModel` |
| `-PPOaug` | PPO with symmetry data augmentation | rsl_rl `Symmetry` extension with `symmloco.tasks.velocity.mdp:mirror_batch` |
| `-PPOeqic` | PPO with an equivariant actor-critic | rsl_rl `EquivariantMLPModel` and `EquivariantGaussianDistribution` (`extras` branch), `EquivariantOnPolicyRunner` |

The environment, rewards and PPO hyperparameters are mjlab's `Mjlab-Velocity-Flat-Unitree-G1`
unchanged; only the model, distribution and symmetry settings differ between the three
methods. With PPOeqic the actor mean is equivariant and the critic invariant by
construction, mirrored joints share their exploration noise, and the observation
normalizer keeps symmetric statistics.

## Dependencies

- mjlab is pinned to commit `936f7a1`.
- `rsl-rl-lib` is overridden (`[tool.uv] override-dependencies`) with the `extras` branch of
  rsl_rl at commit `0f0aa6b`, which provides the equivariant model. It has the same version
  as the PyPI release mjlab requires.

## Tasks

| task id | robot | terrain |
|---|---|---|
| `VelocityFlat-G1-{PPO,PPOaug,PPOeqic}` | Unitree G1 (29 DoF) | flat |

The tasks are registered through the `mjlab.tasks` entry point, so mjlab's own CLIs see them:

```bash
uv run train VelocityFlat-G1-PPOeqic --env.scene.num-envs 4096
uv run play VelocityFlat-G1-PPOeqic --wandb-run-path suz_team/symmloco/<run_id>
```

`play` loads the latest `model_<it>.pt` uploaded to that W&B run (`--wandb-checkpoint-name`
picks another one); `<run_id>` is the id at the end of the run's URL.

## Symmetry

The reflection mirrors the world through the xz-plane and swaps the robot's left and right
sides; every observation term and the action vector transform as a signed permutation
`x -> sign * x[perm]`. `tasks/velocity/mdp/symmetry.py` declares how each observation term transforms
(`TERM_KINDS`, keyed by term name) and derives the joint permutation from the joint names:
`left_*` and `right_*` joints swap, roll and yaw joints change sign. The permutations are
assembled from the live observation manager, so they always match the observation layout;
a term without an entry in `TERM_KINDS` raises an error.

## Layout

Mirrors mjlab's package layout.

```
src/symmloco/
  rl/config.py              SymmlocoPpoAlgorithmCfg (PPO plus symmetry_cfg), EquivariantOnPolicyRunnerCfg
  rl/runner.py              EquivariantOnPolicyRunner: injects the task's reflections into the models
  tasks/velocity/
    mdp/symmetry.py         term kinds, joint/observation/action reflections, mirror_batch
    config/g1/rl_cfg.py     one runner config per method
    config/g1/__init__.py   registers VelocityFlat-G1-{PPO,PPOaug,PPOeqic}
```

PPO and PPOaug run with mjlab's `VelocityOnPolicyRunner`. PPOeqic runs with
`EquivariantOnPolicyRunner`, which is task-agnostic: it calls the functions named by
`observation_reflection_func` and `action_reflection_func` in its config. All three use
mjlab's `unitree_g1_flat_env_cfg`.
