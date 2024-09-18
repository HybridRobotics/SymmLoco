# Leveraging Symmetry in Quadrupedal Dribbling Task

# Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training a Model](#training)
4. [Play a trained policy](#playing)
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

This branch provides the environment used to train go1 to dribble a soccer. The training process uses three different methods: the vanilla PPO (`mlp`), PPO with data augmentation (`aug`), PPO with equivariant / invariant networks (`emlp`).

The code is modified from [dribblebot](https://github.com/Improbable-AI/dribblebot) and [MorphoSymm](https://github.com/Danfoa/MorphoSymm).

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

### Install the `dribblebot` and `MorphoSymm` package

In `SymmLoco/dribblebot`, run `pip install -e .`

In `SymmLoco/MorphoSymm`, run `pip install -e .`

## Training a model <a name="training"></a>

In `SymmLoco/dribblebot`, run 

```bash
python scripts/train_dribbling.py --task=emlp --headless
```

- The trained policy is saved in `SymmLoco/dribblebot/tmp/<method_name>/<date_time>/ac_weights_<iteration>.pt`.
- The following command line arguments are supported:
    - `--headless`: To run headless (no rendering).
    - `--task`: Task name. Choose from ['mlp', 'aug', 'emlp']
    - `--device`: The device used for simulation and RL training. Default is 'cuda:0'.


## Play a trained policy <a name="playing"></a>

We provide a trained policy for each method. They are saved in `SymmLoco/dribblebot/tmp/`. Play the trained `emlp` policy by running:

```bash
python scripts/play_dribbling.py --method=emlp --load_run=2024-09-16-21-10-18
```

You should see a robot dribbling a yellow soccer towards the `x` direction.

The following command line arguments are supported:
- `--headless`: To run headless (no rendering).
- `--task`: Task name. Choose from ['mlp', 'aug', 'emlp']
- `--device`: The device used for simulation and RL training. Default is 'cuda:0'.
- `--checkpoint`: The loaded checkpoint. Default is 'latest'.
- `--load_run`: The loaded run name.

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
