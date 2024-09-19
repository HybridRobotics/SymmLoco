from legged_gym.envs.cyberdog2.c2_common_config import CyberCommonCfg, CyberCommonCfgPPO
import numpy as np
from isaacgym import gymapi

class CyberPushDoorConfig(CyberCommonCfg):
    mode = "train"
    class env(CyberCommonCfg.env):
        num_state_history = 5
        num_single_state = 43

        num_observations = num_state_history * num_single_state
        priv_obs_friction = True
        priv_obs_restitution = True
        priv_obs_joint_friction = True
        priv_obs_contact = True
        priv_obs_com = True
        priv_obs_mass = True
        priv_obs_door_friction = True
        priv_obs_door_dof_vel = True
        num_privileged_obs = num_observations + 3 + 3 \
            + 1 * int(priv_obs_friction)  + 1 * int(priv_obs_restitution) + 12 * int(priv_obs_joint_friction) \
            + 9 * int(priv_obs_contact) + 3 * int(priv_obs_com) + 1 * int(priv_obs_mass) + 1 * int(priv_obs_door_friction) + 1 * int(priv_obs_door_dof_vel) # add door friction

    class init_state(CyberCommonCfg.init_state):
        pos = [0.0, 0.0, 0.11]
        init_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': -80 / 57.3, 
            'RL_thigh_joint': -80 / 57.3, 
            'FR_thigh_joint': -80 / 57.3, 
            'RR_thigh_joint': -80 / 57.3, 
            'FL_calf_joint': 135 / 57.3,
            'RL_calf_joint': 135 / 57.3,
            'FR_calf_joint': 135 / 57.3,
            'RR_calf_joint': 135 / 57.3,
        }

        # init joint angles range
        init_joint_angles_range = {
            key: [value - 0.1, value + 0.1] for (key, value) in init_joint_angles.items()
        }
        door_init_joint_angles_range = {
            'hinge': [0.0, 0.2]
        }
        randomize_rot = False

    class asset(CyberCommonCfg.asset):
        door_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/door.urdf'
        door_density_range = [200, 250]
        door_angular_damping = 0
        door_linear_damping = 0
        door_max_angular_velocity = 1000
        door_max_linear_velocity = 1000
        door_armature = 0.
        door_thickness = 0.01
        left_or_right = 0 # 0: right, 1: left, None: half left half right
        terminate_after_contacts_on = ["base", "head", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh"]#, "FL_foot", "FR_foot"] # allow calf, add head
        penalize_contacts_on = ["base", "head", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh"]#, "FL_foot", "FR_foot"] # stand
        allow_initial_contacts_on = ["foot", "RL_calf", "RR_calf"]
        max_dof_change = 0.3

    class control(CyberCommonCfg.control):
        stiffness = {'joint': 30.0}
        damping = {'joint': 3.0}
        decimation = 4
        kp_factor_range = [0.8, 1.2]
        kd_factor_range = [0.8, 1.2]
    
    class commands(CyberCommonCfg.commands):
        zero_cmd_threshold = 0.0
        curriculum = False
        discretize = True
        separate_lin_ang = False
        clip_ang_vel = 0.25 * np.pi
        default_gait_freq = 2.5
        # resampling_time = 5
        class ranges(CyberCommonCfg.commands.ranges):
            lin_vel_x = [-0.3, 0.3]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-0.5 * np.pi, 0.5 * np.pi]

    class normalization(CyberCommonCfg.normalization):
        class obs_scales(CyberCommonCfg.normalization.obs_scales):
            dof_vel = 0.
            door_dof_pos = 1.
            door_pos = 1.
            base_pos = 1.
            left_or_right = 1.
            door_normal = 1.
            
    class viewer:
        ref_env = 0
        pos = [-10, 0, 6]  # [m]
        lookat = [-9., 5, 3.]  # [m]

    class rewards(CyberCommonCfg.rewards):
        curriculum = True
        cl_init = 0.4
        cl_step = 0.2
        soft_dof_pos_limit = 0.95
        soft_dof_pos_low = None
        soft_dof_pos_high = None
        soft_torque_limit = 0.5

        tracking_sigma = 0.05
        lift_up_threshold = [0.15, 0.42]
        scale_factor_low = 0.25
        scale_factor_high = 0.35
        foot_target = 0.05
        allow_contact_steps = 30

        upright_vec = [0.2, 0.0, 1.0]
        too_upright_threshold = 0.05
        desired_vel_x = 0.5
        before_move_forward_steps = 80

        class scales(CyberCommonCfg.rewards.scales):
            feet_slip = -0.4
            feet_clearance_cmd_linear = -300
            collision = -2.0
            torque_limits = -0.01
            rear_air = -0.5
            action_rate = -0.03
            action_q_diff = -1.
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            dof_pos_limits = -10
            upright = 1.0
            lift_up_linear = 0.5
            
            foot_twist = -0
            foot_shift = -50

            only_move_forward = 3.
            face_forward = -0.1
            not_too_upright = -0.4

    class domain_rand(CyberCommonCfg.domain_rand):
        joint_friction_range = [0.03, 0.08]
        joint_damping_range = [0.02, 0.06]
        door_joint_friction_range = [0.005, 0.02]
        door_joint_damping_range = [0.01, 0.05]
        randomize_door_joint_props = True
        push_interval_s = 5
        max_push_vel_xy = 0.2
        added_mass_range = [-0.5, 0.5]
        com_displacement_range = [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]]
        door_displace_x_range = [0.4, 0.6]
        door_displace_y_range = [-0.2, 0.2]


class CyberPushDoorCfgPPO(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "push_door_cyber"
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        max_iterations = 20000
        save_interval = 100
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'
    class algorithm( CyberCommonCfgPPO.algorithm ):
        learning_rate = 0.00025

class CyberPushDoorCfgPPOAug(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "push_door_cyber_aug"
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPOAugmented'
        max_iterations = 20000
        save_interval = 100
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'
    class algorithm( CyberCommonCfgPPO.algorithm ):
        learning_rate = 0.00025


class CyberPushDoorCfgPPOEMLP(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "push_door_cyber_emlp"
        policy_class_name = 'ActorCriticSymm'
        algorithm_class_name = 'PPO'
        max_iterations = 20000
        save_interval = 100
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = 'elu'
    class algorithm( CyberCommonCfgPPO.algorithm ):
        learning_rate = 0.00025