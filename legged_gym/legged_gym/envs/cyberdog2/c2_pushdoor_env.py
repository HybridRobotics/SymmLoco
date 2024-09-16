from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.envs.cyberdog2.c2_env import CyberEnv, normalize_range
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, quat_conjugate, quat_mul
import torch
from torch import Tensor
from collections import deque
import numpy as np
import os
from PIL import Image as im

class CyberPushDoorEnv(CyberEnv):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.headless:
            camera_position = gymapi.Vec3(-0.5 + self.env_origins[self.cam_env_id, 0], 0.5 + self.env_origins[self.cam_env_id, 1], 1 + self.env_origins[self.cam_env_id, 2])
            camera_target = gymapi.Vec3(0.5 + self.env_origins[self.cam_env_id, 0], 0. + self.env_origins[self.cam_env_id, 1], 0.5 + self.env_origins[self.cam_env_id, 2])
            self.gym.set_camera_location(self.camera_handle, self.envs[self.cam_env_id], camera_position, camera_target)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.overshoot_buf[:] = 0.
        self.q_diff_buf = torch.abs(self.default_dof_pos.to(self.device) + self.cfg.control.action_scale * actions.to(self.device) - self.dof_pos.to(self.device))    
        
        _switch = np.random.uniform()
        if _switch > self.cfg.control.ratio_delay:
            decimation = self.cfg.control.decimation
        else:
            decimation = np.random.randint(self.cfg.control.decimation_range[0] + 1, self.cfg.control.decimation_range[1] + 1)
        for _ in range(decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.cat((self.torques, torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)), dim=-1)))
            self._apply_external_foot_force()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13
                                                            )[:, self.feet_indices, 7:10]
            self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13
                                                                )[:, self.feet_indices, 10:13]
        self.post_physics_step()
        if self.cfg.record.record:
            image = self.get_camera_image()
            image = im.fromarray(image.astype(np.uint8))
            filename = os.path.join(self.cfg.record.folder, "%d.png" % self.common_step_counter)
            image.save(filename)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_external_foot_force(self):
        force = 0.0 * (torch.norm(self.foot_velocities[:, :, :2], dim=-1) * self.contact_forces[:, self.feet_indices, 2]).unsqueeze(dim=-1)
        direction = -self.foot_velocities[:, :, :2] / torch.clamp(torch.norm(self.foot_velocities[:, :, :2], dim=-1), min=1e-5).unsqueeze(dim=-1)
        external_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        external_forces[:, self.feet_indices, :2] = force * direction
        torque = 0.0 * self.contact_forces[:, self.feet_indices, 2].unsqueeze(dim=-1)
        direction = -self.foot_velocities_ang / torch.clamp(torch.norm(self.foot_velocities_ang, dim=-1), min=1e-5).unsqueeze(dim=-1)
        external_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        external_torques[:, self.feet_indices] = torque * direction
        direction = quat_apply(self.base_quat, to_torch([0., -1., 0.], device=self.device).repeat((self.num_envs, 1)))
        pitch_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)[:, 1]
        pitch_vel[pitch_vel > 0] = 0.
        torque = torch.clamp(0 * -pitch_vel.unsqueeze(dim=1) * torch_rand_float(0.8, 1.2, (self.num_envs, 1), device=self.device), min=-50, max=50)
        mask = self.projected_gravity[:, 2] > -0.1
        external_torques[mask, 0] = (torque * direction)[mask]

        # no external forces and torques for the door
        external_forces = torch.cat((external_forces, torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)), dim=1)
        external_torques = torch.cat((external_torques, torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)), dim=1)

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(external_forces), gymtorch.unwrap_tensor(external_torques), gymapi.ENV_SPACE)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.door_base_pos_in_base_frame[:] = quat_rotate_inverse(self.base_quat, self.door_root_states[:, :3] - self.base_pos)
        self.door_base_quat_in_base_frame[:] = quat_mul(quat_conjugate(self.base_quat), self.door_root_states[:, 3:7])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward_vec[:] = quat_rotate_inverse(self.base_quat, self.forward_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13
                                                              )[:, self.feet_indices, 10:13]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:, self.feet_indices, 0:3]

        self.calf_positions = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:, self.calf_indices, 0:3]
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if self.cfg.mode == "test":
            self.collect_test_stats(env_ids)

        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_dof_pos[:] = self.dof_pos[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        self.init_feet_positions[self.episode_length_buf == 1] = self.foot_positions[self.episode_length_buf == 1]

    def collect_test_stats(self, env_ids):
        for env_id in env_ids:
            if self.env_finish_buffer[env_id] < 11:
                if self.env_finish_buffer[env_id] > 0:
                    self.env_success_buffer[env_id] += torch.logical_and(torch.logical_or(torch.logical_and(self.root_states[env_id, 0] - self.env_origins[env_id, 0] > 2, self.door_dof_pos[env_id, 0] > np.pi / 3), self.door_dof_pos[env_id, 0] > np.pi / 2 - 0.1), self.episode_length_buf[env_id] > 150)
                self.env_finish_buffer[env_id] += 1

        if torch.all(self.env_finish_buffer == 11):
            self.finish_count = (self.env_finish_buffer - 1).sum().item()
            self.success_count = self.env_success_buffer.sum().item()
            print("finish count", self.finish_count)
            print("success count", self.success_count)
            print("success rate", self.success_count / self.finish_count)
            print("\n")

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # only explicitly allow foot contact in these mercy steps
        self.reset_buf = torch.logical_and(
            torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1),
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(self.contact_forces[:, self.allow_initial_contact_indices, :], dim=-1) > 1., dim=1),
                self.episode_length_buf <= self.cfg.rewards.allow_contact_steps
            ))
        )
        position_protect = torch.logical_and(
            self.episode_length_buf > 3, torch.any(torch.logical_or(
            self.dof_pos < self.dof_pos_hard_limits[:, 0] + 5 / 180 * np.pi, 
            self.dof_pos > self.dof_pos_hard_limits[:, 1] - 5 / 180 * np.pi
        ), dim=-1))
        stand_air_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.rewards.allow_contact_steps),
            torch.any(self.foot_positions[:, -2:, 2] > 0.06, dim=-1)
        )
        abrupt_change_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.rewards.allow_contact_steps),
            torch.any(torch.abs(self.dof_pos - self.last_dof_pos) > self.cfg.asset.max_dof_change, dim=-1)
        )
        hand_down_condition = torch.logical_and(
            self.episode_length_buf > 80,
            torch.any(self.foot_positions[:, :2, 2] < 0.02, dim=-1)
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= position_protect
        self.reset_buf |= stand_air_condition
        self.reset_buf |= abrupt_change_condition
        self.reset_buf |= hand_down_condition
        if self.cfg.mode == "play":
            self.reset_buf |= (self.episode_length_buf > 1000)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.actions[env_ids] = 0.

        # update noised_init_door_bottom_corner_pos
        self.noised_init_door_bottom_corner_pos[env_ids, :2] = torch.clone(self.door_root_states[env_ids, :2])
        cosine_tensor = torch.cos(self.door_dof_pos[env_ids])
        sine_tensor = torch.sin(self.door_dof_pos[env_ids])
        offset_left_tensor = torch.tensor([[0.03, 0.025]], device=self.device) + torch.cat([-0.05 * cosine_tensor, 0.05 * sine_tensor], dim=-1)
        offset_right_tensor = torch.tensor([[0.03, -0.025]], device=self.device) + torch.cat([-0.05 * cosine_tensor, -0.05 * sine_tensor], dim=-1)
        offset_tensor = torch.where(self.left_or_right_vec[env_ids] == torch.tensor([0,1], device=self.device), offset_left_tensor, offset_right_tensor)
        self.noised_init_door_bottom_corner_pos[env_ids, :2] += offset_tensor
        self.noised_init_door_bottom_corner_pos[env_ids, :2] += torch.rand((len(env_ids), 2), device=self.device) * 0.04 - 0.02

        # update noised_init_door_normal_vector
        self.noised_init_door_normal_vector[env_ids, :2] = torch.where(self.left_or_right_vec[env_ids] == torch.tensor([0,1], device=self.device), torch.cat([-cosine_tensor, sine_tensor], dim=-1), torch.cat([-cosine_tensor, -sine_tensor], dim=-1))
        self.noised_init_door_normal_vector[env_ids, :2] += torch.rand((len(env_ids), 2), device=self.device) * 0.04 - 0.02
        self.noised_init_door_normal_vector[env_ids, :2] /= torch.norm(self.noised_init_door_normal_vector[env_ids, :2], dim=-1, keepdim=True)

        # randomly set initial phase
        self.phase[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device)

    def _compute_common_obs(self):
        phase_input = self.phase.clone()
        phase_input[self.phase > 0.5] = 0.5 - phase_input[self.phase > 0.5]
        common_obs_buf = torch.cat([self.projected_gravity,
                                    self.projected_forward_vec,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.actions,
                                    phase_input * 2,
                                    -phase_input * 2,
                                    (self.base_pos - self.env_origins) * self.obs_scales.base_pos,
                                    (self.noised_init_door_bottom_corner_pos - self.env_origins) * self.obs_scales.door_pos,
                                    self.noised_init_door_normal_vector * self.obs_scales.door_normal,
                                    self.left_or_right_vec * self.obs_scales.left_or_right,
        ],dim=-1)

        return common_obs_buf
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_state, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        start_index = 0
        noise_vec[start_index:start_index + 3] = noise_scales.gravity * noise_level
        noise_vec[start_index + 3: start_index + 6] = noise_scales.gravity * noise_level
        start_index += 6
        noise_vec[start_index:start_index + 12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[start_index + 12:start_index + 24] = 0. # previous actions
        noise_vec[start_index + 24: start_index + 26] = 0. # clock input
        start_index = start_index + 26
        noise_vec[start_index:start_index + 3] = 0.05 * noise_level # base position
        noise_vec[start_index + 3: start_index + 6] = 0. # door init position, noise has been added before
        noise_vec[start_index + 6: start_index + 9] = 0. # door init normal vector, noise has been added before
        noise_vec[start_index + 9: start_index + 11] = 0.
        start_index += 11

        assert start_index == self.cfg.env.num_single_state
        return noise_vec
    
    def _compute_privileged_obs(self):
        if self.cfg.env.num_privileged_obs is not None:
            privileged_obs = self.obs_buf.clone()
            if not self.cfg.env.obs_base_vel:
                privileged_obs = torch.cat([privileged_obs, self.base_lin_vel * self.obs_scales.lin_vel], dim=-1)
            if not self.cfg.env.obs_base_vela:
                privileged_obs = torch.cat([privileged_obs, self.base_ang_vel * self.obs_scales.ang_vel], dim=-1)
            if self.cfg.env.priv_obs_friction:
                privileged_obs = torch.cat([privileged_obs, normalize_range(self.friction_coeffs.unsqueeze(dim=-1), self.cfg.domain_rand.friction_range)], dim=-1)
            if self.cfg.env.priv_obs_restitution:
                privileged_obs = torch.cat([privileged_obs, normalize_range(self.restitution_coeffs.unsqueeze(dim=-1), self.cfg.domain_rand.restitution_range)], dim=-1)
            if self.cfg.env.priv_obs_joint_friction:
                privileged_obs = torch.cat([privileged_obs, normalize_range(self.joint_friction, self.joint_friction_range)], dim=-1)
            if self.cfg.env.priv_obs_com:
                privileged_obs = torch.cat([privileged_obs, normalize_range(self.com_displacement, self.cfg.domain_rand.com_displacement_range)], dim=-1)
            if self.cfg.env.priv_obs_mass:
                privileged_obs = torch.cat([privileged_obs, normalize_range(self.mass_offset.unsqueeze(dim=-1), self.cfg.domain_rand.added_mass_range)], dim=-1)
            if self.cfg.env.priv_obs_contact:
                link_in_contact = (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float()
                privileged_obs = torch.cat([privileged_obs, link_in_contact], dim=-1)
            if self.cfg.env.priv_obs_door_friction:
                privileged_obs = torch.cat([privileged_obs, self.door_joint_friction], dim=-1)
            if self.cfg.env.priv_obs_door_dof_vel:
                privileged_obs = torch.cat([privileged_obs, self.door_dof_vel], dim=-1)
            self.privileged_obs_buf[:] = privileged_obs

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        door_asset_path = self.cfg.asset.door_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        door_asset_root = os.path.dirname(door_asset_path)
        door_asset_file = os.path.basename(door_asset_path)

        door_asset_options = gymapi.AssetOptions()
        door_asset_options.collapse_fixed_joints = True
        door_asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        door_asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        door_asset_options.fix_base_link = True
        door_asset_options.angular_damping = self.cfg.asset.door_angular_damping
        door_asset_options.linear_damping = self.cfg.asset.door_linear_damping
        door_asset_options.max_angular_velocity = self.cfg.asset.door_max_angular_velocity
        door_asset_options.max_linear_velocity = self.cfg.asset.door_max_linear_velocity
        door_asset_options.armature = self.cfg.asset.door_armature
        door_asset_options.thickness = self.cfg.asset.door_thickness
        door_asset_options.disable_gravity = self.cfg.asset.disable_gravity
        door_asset_options.override_com = True
        door_asset_options.override_inertia = True

        door_asset = self.gym.load_asset(self.sim, door_asset_root, door_asset_file, door_asset_options)
        
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.door_num_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        door_dof_props_asset = self.gym.get_asset_dof_properties(door_asset)
        door_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(door_asset)
        self.all_num_bodies = self.num_bodies + self.door_num_bodies

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s] # FL, FR, RL, RR
        calf_names = [s for s in body_names if "calf" in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name == s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        allow_initial_contact_names = []
        for name in self.cfg.asset.allow_initial_contacts_on:
            allow_initial_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.num_actors = 2
        self.joint_friction = []
        self.door_joint_friction = []
        self.joint_damping_range = self.cfg.domain_rand.joint_damping_range
        self.joint_friction_range = self.cfg.domain_rand.joint_friction_range
        self.door_joint_damping_range = self.cfg.domain_rand.door_joint_damping_range
        self.door_joint_friction_range = self.cfg.domain_rand.door_joint_friction_range
        self.com_displacement = []
        self.mass_offset = []
        for i in range(self.num_envs):
            door_asset_options.density = np.random.uniform(*self.cfg.asset.door_density_range)
            door_asset = self.gym.load_asset(self.sim, door_asset_root, door_asset_file, door_asset_options)
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            door_rigid_shape_props = door_rigid_shape_props_asset.copy()
            for s in range(len(door_rigid_shape_props)):
                door_rigid_shape_props[s].friction = self.friction_coeffs[i]
                door_rigid_shape_props[s].restitution = self.restitution_coeffs[i]
            self.gym.set_asset_rigid_shape_properties(door_asset, door_rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            door_handle = self.gym.create_actor(env_handle, door_asset, start_pose, "door", i, self.cfg.asset.self_collisions, 0) # TODO!!!
            dof_props = self._process_dof_props(dof_props_asset, i)
            door_dof_props = door_dof_props_asset.copy()
            # randomize?
            if self.cfg.domain_rand.randomize_joint_props:
                for j in range(len(dof_props)):
                    dof_props["damping"][j] = np.random.uniform(*self.joint_damping_range)
                    dof_props["friction"][j] = np.random.uniform(*self.joint_friction_range)
            if self.cfg.domain_rand.randomize_door_joint_props:
                for j in range(len(door_dof_props)):
                    door_dof_props["damping"][j] = np.random.uniform(*self.door_joint_damping_range)
                    door_dof_props["friction"][j] = np.random.uniform(*self.door_joint_friction_range)
            if i == self.cam_env_id:
                print("joint props")
                for j in range(len(dof_props)):
                    print(j, "damping", dof_props["damping"][j], "friction", dof_props["friction"][j])
                print("door joint props")
                for j in range(len(door_dof_props)):
                    print(j, "damping", door_dof_props["damping"][j], "friction", door_dof_props["friction"][j])
            self.joint_friction.append(np.array([dof_props["friction"][j] for j in range(len(dof_props))]))
            self.door_joint_friction.append(np.array([door_dof_props["friction"][j] for j in range(len(door_dof_props))]))
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.set_actor_dof_properties(env_handle, door_handle, door_dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.joint_friction = torch.from_numpy(np.array(self.joint_friction)).float().to(self.device)
        self.door_joint_friction = torch.from_numpy(np.array(self.door_joint_friction)).float().to(self.device)
        if self.cfg.domain_rand.randomize_base_mass:
            self.mass_offset = torch.from_numpy(np.array(self.mass_offset)).float().to(self.device)
        else:
            self.mass_offset = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch.from_numpy(np.array(self.com_displacement)).float().to(self.device)
        else:
            self.com_displacement = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], calf_names[i])
        
        link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.feet_link_index = torch.tensor([link_dict[_name] for _name in feet_names]).to(dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        self.base_contact_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "base")
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        self.allow_initial_contact_indices = torch.zeros(len(allow_initial_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(allow_initial_contact_names)):
            self.allow_initial_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], allow_initial_contact_names[i])

    def _reset_robot_states(self, env_ids):
        self._reset_dofs_rand(env_ids)
        self._reset_root_states_rand(env_ids)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7] -= torch.rand(self.num_envs, device=self.device) * max_vel # lin vel along -x
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _reset_dofs_rand(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.init_dof_pos_range[:, 0] + torch_rand_float(0., 1., (len(env_ids), self.num_dof), device=self.device) * (self.init_dof_pos_range[:, 1] - self.init_dof_pos_range[:, 0])
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.door_dof_pos[env_ids] = self.cfg.init_state.door_init_joint_angles_range['hinge'][0] + torch_rand_float(0., 1., (len(env_ids), 1), device=self.device) * (self.cfg.init_state.door_init_joint_angles_range['hinge'][1] - self.cfg.init_state.door_init_joint_angles_range['hinge'][0])
        self.door_dof_vel[env_ids] = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device)

        # TODO: Important! should feed actor id, not env id
        env_ids_int32 = torch.cat((self.num_actors * env_ids.to(dtype=torch.int32), self.num_actors * env_ids.to(dtype=torch.int32) + 1))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.all_dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_rand(self, env_ids): #changed!
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # False
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.init_state.randomize_rot: # False
            rand_rpy = torch_rand_float(-np.pi*15/180.0, np.pi*15/180.0, (len(env_ids), 3), device=self.device) #参数：rand在正负15度
            rand_rpy = rand_rpy + torch.Tensor(get_euler_xyz(self.base_init_state[3:7].unsqueeze(0))).to(self.device)
            self.root_states[env_ids, 3: 7] = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])  #!!!changed to +=
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        # door 
        if self.cfg.asset.left_or_right == None: #half right half left
            for env_id in env_ids:
                init_door_displace_x = np.random.uniform(*self.cfg.domain_rand.door_displace_x_range)
                init_door_displace_y = np.random.uniform(*self.cfg.domain_rand.door_displace_y_range)
                if env_id % 2 == 0: # right
                    self.door_root_states[env_id, :2] = self.root_states[env_id, :2] + torch.tensor([init_door_displace_x, 0.45 + init_door_displace_y], device=self.device, dtype=torch.float, requires_grad=False)
                    self.door_root_states[env_id, 2] = 1.42
                    self.door_root_states[env_id, 3:7] = torch.tensor([0., 0., -0.7071, 0.7071], device=self.device, dtype=torch.float, requires_grad=False)
                else: # left
                    self.door_root_states[env_id, :2] = self.root_states[env_id, :2] + torch.tensor([init_door_displace_x, -0.45 + init_door_displace_y], device=self.device, dtype=torch.float, requires_grad=False)
                    self.door_root_states[env_id, 2] = 1.42
                    self.door_root_states[env_id, 3:7] = torch.tensor([0.7071, 0.7071, 0., 0.], device=self.device, dtype=torch.float, requires_grad=False)
        elif self.cfg.asset.left_or_right == 0: # all right
            init_door_displace_x = np.random.uniform(*self.cfg.domain_rand.door_displace_x_range)
            init_door_displace_y = np.random.uniform(*self.cfg.domain_rand.door_displace_y_range)
            self.door_root_states[env_ids, :2] = self.root_states[env_ids, :2] + torch.tensor([[init_door_displace_x, 0.45 + init_door_displace_y]], device=self.device, dtype=torch.float, requires_grad=False)
            self.door_root_states[env_ids, 2] = 1.42
            self.door_root_states[env_ids, 3:7] = torch.tensor([[0., 0., -0.7071, 0.7071]], device=self.device, dtype=torch.float, requires_grad=False)
        else: # all left
            init_door_displace_x = np.random.uniform(*self.cfg.domain_rand.door_displace_x_range)
            init_door_displace_y = np.random.uniform(*self.cfg.domain_rand.door_displace_y_range)
            self.door_root_states[env_ids, :2] = self.root_states[env_ids, :2] + torch.tensor([[init_door_displace_x, -0.45 + init_door_displace_y]], device=self.device, dtype=torch.float, requires_grad=False)
            self.door_root_states[env_ids, 2] = 1.42
            self.door_root_states[env_ids, 3:7] = torch.tensor([[0.7071, 0.7071, 0., 0.]], device=self.device, dtype=torch.float, requires_grad=False)

        env_ids_int32 = torch.cat((self.num_actors * env_ids.to(dtype=torch.int32), self.num_actors * env_ids.to(dtype=torch.int32) + 1))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :]
        self.door_root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 1, :]
        self.all_dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_state.view(self.num_envs, -1, 2)[:, :-1, :]
        self.door_dof_state = self.all_dof_state.view(self.num_envs, -1, 2)[:, -1, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.door_dof_pos = self.door_dof_state.view(self.num_envs, 1, 2)[..., 0]
        self.door_dof_vel = self.door_dof_state.view(self.num_envs, 1, 2)[..., 1]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.door_base_pos_in_base_frame = quat_rotate_inverse(self.base_quat, self.door_root_states[:, :3] - self.base_pos)
        self.door_base_quat_in_base_frame = quat_mul(quat_conjugate(self.base_quat), self.door_root_states[:, 3:7])
        self.lag_buffer = torch.zeros((self.num_envs, self.num_dof, self.cfg.domain_rand.lag_timesteps + 1), dtype=torch.float, device=self.device)
        self.lag_steps = torch.zeros((self.num_envs, self.num_dof), dtype=int, device=self.device)
        self.valid_history_length = torch.ones((self.num_envs, self.num_dof), dtype=int, device=self.device) * (self.cfg.domain_rand.lag_timesteps + 1)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, all_num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # initialize some data used later on
        self.left_or_right_vec = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.asset.left_or_right == None:
            self.left_or_right_vec[1::2, 1] = 1.
            self.left_or_right_vec[::2, 0] = 1.
        else:
            self.left_or_right_vec[:, self.cfg.asset.left_or_right] = 1.
        
        self.noised_init_door_bottom_corner_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.noised_init_door_normal_vector = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # TODO: only for cyber dog
        if hasattr(self.cfg.rewards, "upright_vec"):
            self.upright_vec = to_torch(self.cfg.rewards.upright_vec, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.lagged_actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        if self.cfg.commands.num_commands > 4:
            self.commands_gait_mean = torch.tensor([
                np.mean(self.command_ranges["base_height"]), 
                np.mean(self.command_ranges["foot_height"]),
                np.mean(self.command_ranges["frequency"]),
                0., 0., 0.
            ], device=self.device, requires_grad=False).float()
            self.commands_gait_scale = torch.tensor([
                (self.command_ranges["base_height"][1] - self.command_ranges["base_height"][0]) / 2,
                (self.command_ranges["foot_height"][1] - self.command_ranges["foot_height"][0]) / 2,
                (self.command_ranges["frequency"][1] - self.command_ranges["frequency"][0]) / 2,
                1., 1., 1.
            ], device=self.device, requires_grad=False).float()
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward_vec = quat_rotate_inverse(self.base_quat, self.forward_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:, self.feet_indices, 10: 13]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:, self.feet_indices, 0:3]
        self.calf_positions = self.rigid_body_state.view(self.num_envs, self.all_num_bodies, 13)[:, self.calf_indices, 0:3]
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.reset_states_buffer = None
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos_range = torch.zeros((self.num_dof, 2), dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            init_angle = self.cfg.init_state.init_joint_angles[name]
            self.init_dof_pos[i] = init_angle
            self.init_dof_pos_range[i][0] = self.cfg.init_state.init_joint_angles_range[name][0]
            self.init_dof_pos_range[i][1] = self.cfg.init_state.init_joint_angles_range[name][1]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)
        self.overshoot_buf = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.q_diff_buf = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # motor target and cur q
        if hasattr(self.cfg.init_state, "HIP_OFFSETS"):
            self.HIP_OFFSETS = torch.from_numpy(self.cfg.init_state.HIP_OFFSETS).to(dtype=torch.float, device=self.device)  # (4, 3)
        if hasattr(self.cfg.init_state, "DEFAULT_HIP_POSITIONS"):
            self.DEFAULT_HIP_POSITIONS = torch.from_numpy(np.array(self.cfg.init_state.DEFAULT_HIP_POSITIONS)).to(dtype=torch.float, device=self.device)
        self.save_data_buffer = {"q": [], "q_des": [], "projected_gravity": []}
        if self.cfg.record.record:
            if self.cfg.commands.heading_command:
                self.all_commands = [
                    torch.from_numpy(np.array([0., self.command_ranges["lin_vel_y"][1], 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., 0., -np.pi / 2])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., 0., np.pi / 2])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                ]
                self.all_commands = []
            else:
                self.all_commands = [
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][0], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                ]
        self.num_history = self.cfg.env.num_state_history
        self.num_stacked_obs = self.cfg.env.num_stacked_obs # The common obs in RMA
        self.obs_history = deque(maxlen=self.cfg.env.num_state_history)
        for _ in range(self.cfg.env.num_state_history):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_state, dtype=torch.float, device=self.device))
        self.num_env_factors = self.cfg.env.num_env_factors
        self.env_factor_buf = torch.zeros((self.num_envs, self.num_env_factors), dtype=torch.float, device=self.device)
        self.init_feet_positions = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device)
        
        if self.cfg.mode == "test":
            self.finish_count = 0
            self.success_count = 0
            self.env_finish_buffer = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)
            self.env_success_buffer = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)

    def _step_contact_targets(self):
        frequencies = self.cfg.commands.default_gait_freq
        self.phase = torch.remainder(self.phase + self.dt * frequencies, 1.0)

    def _reward_feet_slip(self):
        condition = self.foot_positions[:, :, 2] < 0.03
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(condition.float() * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip
    
    def _reward_feet_clearance_cmd_linear(self):
        foot_height = (self.foot_positions[:, -2:, 2]).view(self.num_envs, -1)
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, -2:, :2])
        target_height = self.cfg.rewards.foot_target * torch.clip(torch.cat((torch.sin(2 * np.pi * self.phase), -torch.sin(2 * np.pi * self.phase)), dim = 1), min = 0) + terrain_at_foot_height + 0.02 
        rew_foot_clearance = torch.square(target_height - foot_height)
        condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_lift_up_linear(self):
        root_height = self.root_states[:, 2]
        reward = (root_height - self.cfg.rewards.lift_up_threshold[0]) / (self.cfg.rewards.lift_up_threshold[1] - self.cfg.rewards.lift_up_threshold[0])
        reward = torch.clamp(reward, 0., 1.)
        return reward

    def _reward_rear_air(self):
        contact = self.contact_forces[:, self.feet_indices[-2:], 2] < 1.
        calf_contact = self.contact_forces[:, self.calf_indices[-2:], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        reward = torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
        return reward
    
    def _reward_foot_twist(self):
        vxy = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        vang = torch.norm(self.foot_velocities_ang, dim=-1)
        condition = self.foot_positions[:, :, 2] < 0.025
        reward = torch.mean((vxy + 0.1 * vang) * condition.float(), dim=1)
        return reward
    
    def _reward_foot_shift(self):
        desired_foot_positions = torch.clone(self.init_feet_positions[:, 2:])
        desired_foot_positions[:, :, 2] = 0.02
        rear_foot_shift = torch.norm(self.foot_positions[:, 2:] - desired_foot_positions, dim=-1).mean(dim=1)
        init_ffoot_positions = torch.clone(self.init_feet_positions[:, :2])
        front_foot_shift = torch.norm( torch.stack([
            (init_ffoot_positions[:, :, 0] - self.foot_positions[:, :2, 0]).clamp(min=0), 
            torch.abs(init_ffoot_positions[:, :, 1] - self.foot_positions[:, :2, 1])
        ], dim=-1), dim=-1).mean(dim=1)
        condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        reward = (front_foot_shift + rear_foot_shift) * condition.float()
        return reward
    
    def _reward_only_move_forward(self):
        desired_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        desired_vel[:, 0] = self.cfg.rewards.desired_vel_x
        lin_vel_error = torch.sum(torch.square(self.root_states[:, 7:10] - desired_vel), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        condition = self.root_states[:, 2] > 0.35
        condition2 = self.episode_length_buf > self.cfg.rewards.before_move_forward_steps
        return reward * condition.float() * condition2.float()

    def _reward_face_forward(self):
        heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        reward = 1 - torch.cos(torch.atan2(heading_vec[:, 1], heading_vec[:, 0]))
        return reward
    
    def _reward_not_too_upright(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        vertical = to_torch([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))
        cosine_dist = torch.sum(forward * vertical, dim=-1)
        cond = torch.acos(cosine_dist) < self.cfg.rewards.too_upright_threshold
        reward = cond.float() * 1
        return reward