# Leveraging Symmetry in RL-based Legged Locomotion Control

# Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training a Model](#training)
4. [Playing a trained policy](#playing)
5. [Bibtex](#bibtex)

## Overview <a name="overview"></a>

This repository provides an implementation of the paper:

<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://suz-tsinghua.github.io/SymmLoco-page/" target="_blank">
      <b> Leveraging Symmetry in RL-based Legged Locomotion Control </b>
      </a>
      <br>
      <em>IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)</em>, 2024
      <br>
      <a href="https://arxiv.org/abs/2403.17320">paper</a> /
      <a href="https://suz-tsinghua.github.io/SymmLoco-page/" target="_blank">project page</a>
    <br>
</td>

<br>

This repository provides the environment used to train cyberdog2 to perform four tasks: Door Pushing, Stand Turning (Stand Dancing), Slope Walking and Dribbling. This branch implements Door Pushing, Stand Turning and Slope Walking, while the branch `dribbling` implements the Dribbling task. The training process uses three different methods: the vanilla PPO (`mlp`), PPO with data augmentation (`aug`), PPO with equivariant / invariant networks (`emlp`).

The code is modified from Isaac Gym Environments for Legged Robots and based on [legged_stand_dance](https://github.com/IrisLi17/legged_stand_dance) and [MorphoSymm](https://github.com/Danfoa/MorphoSymm).

## Installation <a name="installation"></a>

### Create a new conda environment with Python (3.8 suggested)

```bash
conda create -n symmloco python==3.8
conda activate symmloco
```

### Install pytorch 1.10 with cuda-11.3:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

### Install the `MorphoSymm`, `rsl_rl` and `legged_gym` packages

In `SymmLoco/MorphoSymm`, run `pip install -e .`

In `SymmLoco/rsl_rl`, run `pip install -e .`

In `SymmLoco/legged_gym`, run `pip install -e .`


### Install other dependencies 
```bash
pip install tensorboard wandb
```

## Training a model <a name="training"></a>
In `SymmLoco/legged_gym`, run

```bash
python legged_gym/scripts/train.py --task=cyber2_push_door_emlp --headless --right
```

-  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
-  To run headless (no rendering) use `--headless`.
- **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in `legged_gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
-  The following command line arguments override the values set in the config files:
    - --task TASK: Task name. All supported tasks can be found in `legged_gym/scripts/train.py`.
    - --resume:   Resume training from a checkpoint
    - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
    - --run_name RUN_NAME:  Name of the run.
    - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
    - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
    - --num_envs NUM_ENVS:  Number of environments to create.
    - --seed SEED:  Random seed.
    - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
    - --left: Set the door open direction to `left`. If not set, half number of doors are set to left and half are set to right.
    - --right: Set the door open direction to `right`. If not set, half number of doors are set to left and half are set to right.

### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`

## Playing a trained policy <a name="playing"></a>
We provide a checkpoint of Door Pushing task trained by `emlp`. Play the trained policy by running:

```bash
python legged_gym/scripts/play.py --task=cyber2_push_door_emlp --load_run=2024-09-17-23-10-31_ --checkpoint=20000
```

You should see a quadrupedal robot standing up on its rear legs and pushing the door open.

## Bibtex <a name="bibtex"></a>

```
@inproceedings{su2024leveraging,
    title={Leveraging Symmetry in RL-based Legged Locomotion Control},
    author={Su, Zhi and Huang, Xiaoyu and Ordoñez-Apraez, Daniel and Li, Yunfei and Li, Zhongyu and Liao, Qiayuan and Turrisi, Giulio and Pontil, Massimiliano and Semini, Claudio and Wu, Yi and Sreenath, Koushil},
    booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2024},
    organization={IEEE}
}
```