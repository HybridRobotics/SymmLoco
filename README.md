# Leveraging Symmetry in RL-based Legged Locomotion Control
This repository provides the environment used to train cyberdog2 to perform three tasks: Door Pushing, Stand Turning (Stand Dancing), Slope Walking. The training process uses three different methods: the vanilla PPO, PPO with data augmentation, PPO with equivariant / invariant networks.

The code is modified from Isaac Gym Environments for Legged Robots and based on [legged_stand_dance](https://github.com/IrisLi17/legged_stand_dance).

### Useful Links ###
Project website: https://suz-tsinghua.github.io/SymmLoco-page/
Paper: https://arxiv.org/abs/2403.17320

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install MorphoSymm, modified rsl_rl (PPO implementation) and legged_gym
   -  `cd MorphoSymm && pip install -e .`
   -  `cd rsl_rl && pip install -e .` 
   -  `cd legged_gym && pip install -e .`
5. Install other dependencies 
    - pip install tensorboard wandb

### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=cyber2_push_door_emlp --headless --right```
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
2. Play a trained policy:  
    TODO
<!-- ```python legged_gym/scripts/play.py --task=cyber2_stand_dance --load_run 2023-09-01-12-57-42_initsit_sliph0.03-0.4_qdiff-1_shift-50_vec0.2_resetpos5deg_initcontact30_com0.01_wvel --checkpoint 18000 --headless```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config. -->

### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`
