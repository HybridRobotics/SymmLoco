from isaacgym import gymtorch, gymapi
import torch
from collections import deque
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, to_torch, torch_rand_float
from legged_gym.utils.math import quat_apply_yaw
import numpy as np

class StackObsEnv(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.num_history = self.cfg.env.num_state_history
        self.num_stacked_obs = self.cfg.env.num_stacked_obs # The common obs in RMA
        self.obs_history = deque(maxlen=self.cfg.env.num_state_history)
        for _ in range(self.cfg.env.num_state_history):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_state, dtype=torch.float, device=self.device))
        self.num_env_factors = self.cfg.env.num_env_factors
        self.env_factor_buf = torch.zeros((self.num_envs, self.num_env_factors), dtype=torch.float, device=self.device)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] = 0.

class CyberEnv(StackObsEnv):
    def compute_observations(self):
        cur_obs_buf = self._compute_common_obs()
        # add noise if needed
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec
        self.obs_history.append(cur_obs_buf)
        self.obs_buf = torch.cat([self.obs_history[i] for i in range(len(self.obs_history))], dim=-1)
        self._compute_privileged_obs()
    
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
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(external_forces), gymtorch.unwrap_tensor(external_torques), gymapi.ENV_SPACE)

    def _compute_common_obs(self):
        raise NotImplementedError
    
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
            self.privileged_obs_buf[:] = privileged_obs
    
    def _reward_upright(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
        # dot product with [0, 0, 1]
        # cosine_dist = forward[:, 2]
        reward = torch.square(0.5 * cosine_dist + 0.5)
        return reward
    
    def _reward_lift_up(self):
        root_height = self.root_states[:, 2]
        # four leg stand is ~0.28
        # sit height is ~0.385
        reward = torch.exp(root_height - self.cfg.rewards.lift_up_threshold) - 1
        return reward

    def _reward_collision(self):
        reward = super()._reward_collision()
        cond = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        reward = reward * cond.float()
        return reward
    
    def _reward_action_q_diff(self):
        condition = self.episode_length_buf <= self.cfg.rewards.allow_contact_steps
        reward = torch.sum(torch.square(self.q_diff_buf), dim=-1) * condition.float()
        return reward
    
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip

def normalize_range(x: torch.Tensor, limit):
    if isinstance(limit[0], list):
        low = torch.from_numpy(np.array(limit[0])).to(x.device)
        high = torch.from_numpy(np.array(limit[1])).to(x.device)
    else:
        low = limit[0]
        high = limit[1]
    mean = (low + high) / 2
    scale = (high - low) / 2
    if isinstance(scale, torch.Tensor):
        scale = torch.clamp(scale, min=1e-5)
    else:
        scale = max(scale, 1e-5)
    return (x - mean) / scale