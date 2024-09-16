# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger
import os
import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.task == "go1_highlevel":
        low_env_cfg = env_cfg.low_env
    else:
        low_env_cfg = env_cfg
    low_env_cfg.env.num_envs = 2000
    low_env_cfg.record.record = RECORD_FRAMES
    low_env_cfg.record.folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    low_env_cfg.terrain.curriculum = False
    low_env_cfg.rewards.curriculum = False
    low_env_cfg.mode = "test"
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.com_displacement_range = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    if "stand_dance" in args.task:
        low_env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [0.5 * np.pi, 0.5 * np.pi]
    elif "push_door" in args.task:
        # For ood test
        # low_env_cfg.init_state.randomize_rot = True
        if args.left:
            low_env_cfg.asset.left_or_right = 1 # 0: right, 1: left
        elif args.right:
            low_env_cfg.asset.left_or_right = 0
    elif "walk_slope" in args.task:
        low_env_cfg.terrain.curriculum = True
        low_env_cfg.terrain.max_init_terrain_level = 5
        low_env_cfg.commands.ranges.lin_vel_x = [0.3, 0.3]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [0., 0.]

    # if os.path.exists(low_env_cfg.record.folder):
    #     shutil.rmtree(low_env_cfg.record.folder)
    # os.makedirs(low_env_cfg.record.folder, exist_ok=True)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, is_highlevel=(args.task == "go1_highlevel"))
    low_env = env.low_level_env if args.task == "go1_highlevel" else env
    # obs = env.get_observations()
    obs, *_ = env.reset()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # currently, both high and low level shares the same number of obs
        if hasattr(ppo_runner.alg.actor_critic, "adaptation_module"):
            input_dim = env.num_obs * env.num_history + env.num_obs * env.num_stacked_obs
        else:
            input_dim = env.num_obs
        export_policy_as_onnx(ppo_runner.alg.actor_critic, input_dim, path)
        print('Exported policy as jit script to: ', path)
    # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    # torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), os.path.join(path, 'policy.pt'))
    # print('Exported policy to: ', os.path.join(path, 'policy.pt'))

    logger = Logger(low_env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 100 # number of steps before print average episode rewards
    camera_position = np.array(low_env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(low_env_cfg.viewer.lookat) - np.array(low_env_cfg.viewer.pos)
    img_idx = 0

    total_steps = int(1000000000 * env.max_episode_length) #used to be 5

    episode_reward_tmp = 0
    episode_length_tmp = 0
    episode_reward_buf = []
    episode_length_buf = []
    
    for i in range(total_steps): 
        with torch.no_grad():
            actions = policy(obs)

        obs, _, rews, dones, infos = env.step(actions.detach())
        episode_reward_tmp += rews
        episode_length_tmp += torch.ones(obs.shape[0], device=obs.device)

        if MOVE_CAMERA:
            camera_position += camera_vel * low_env.dt
            low_env.set_camera(camera_position, camera_position + camera_direction)
        # if  0 < i < stop_rew_log:
        if infos["episode"]:
            num_episodes = torch.sum(low_env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
            episode_reward_buf.extend(episode_reward_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_reward_tmp[low_env.reset_buf] = 0.
            episode_length_buf.extend(episode_length_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_length_tmp[low_env.reset_buf] = 0.
        # elif i==stop_rew_log:
        if torch.all(low_env.env_finish_buffer == 11):
            logger.print_rewards()
            print("Mean episode reward", np.mean(episode_reward_buf), "N episodes", len(episode_reward_buf))
            print("Mean episode length", np.mean(episode_length_buf), "N episodes", len(episode_length_buf))
            break
 
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
