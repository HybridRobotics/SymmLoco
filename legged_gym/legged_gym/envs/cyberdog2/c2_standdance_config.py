from legged_gym.envs.cyberdog2.c2_common_config import CyberCommonCfg, CyberCommonCfgPPO
import numpy as np

use_vel = True
ob_t = False
init_pose = "sit"
class CyberStandDanceConfig(CyberCommonCfg):
    mode = "train"
    class env(CyberCommonCfg.env):
        num_state_history = 3
        num_single_state = 42 + 3 + 2 + 1 * int(ob_t)# add command # add hand targets!!
        num_observations = num_state_history * num_single_state
        priv_obs_friction = True
        priv_obs_restitution = True
        priv_obs_joint_friction = True
        priv_obs_contact = True
        priv_obs_com = True
        priv_obs_mass = True
        num_privileged_obs = num_observations + 3 + 3 \
            + 1 * int(priv_obs_friction)  + 1 * int(priv_obs_restitution) + 12 * int(priv_obs_joint_friction) \
            + 11 * int(priv_obs_contact) + 3 * int(priv_obs_com) + 1 * int(priv_obs_mass)
        vel_cmd = use_vel
        obs_t = ob_t
    
    class init_state(CyberCommonCfg.init_state):
        if init_pose == "stand":
            # stand
            pos = [0.0, 0.0, 0.25] # x,y,z [m]
        elif init_pose == "sit":
            # sit lower
            pos = [0.0, 0.0, 0.11]
        elif init_pose == "upright":
            pos = [0.0, 0.0, 0.39]
            rot = [0.,-np.sin(np.pi / 4),0.,np.cos(np.pi / 4)]
        if init_pose == "stand":
            # stand
            init_joint_angles = {
                'FL_hip_joint': 0.0,
                'RL_hip_joint': 0.0,
                'FR_hip_joint': 0.0,
                'RR_hip_joint': 0.0,
                'FL_thigh_joint': -45 / 57.3, # -80 / 57.3, 
                'RL_thigh_joint': -45 / 57.3, # -80 / 57.3, 
                'FR_thigh_joint': -45 / 57.3, # -80 / 57.3, 
                'RR_thigh_joint': -45 / 57.3, # -80 / 57.3, 
                'FL_calf_joint': 70 / 57.3, # 135 / 57.3,
                'RL_calf_joint': 70 / 57.3, # 135 / 57.3,
                'FR_calf_joint': 70 / 57.3, # 135 / 57.3,
                'RR_calf_joint': 70 / 57.3, # 135 / 57.3,
            }
        elif init_pose == "sit":
            # sit
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
        elif init_pose == "upright":
            # only for creating env
            init_joint_angles = {
                'FL_hip_joint': 0.0727,
                'FL_thigh_joint': -0.3136, 
                'FL_calf_joint': 1.4936,
                'FR_hip_joint': 0.3127,
                'FR_thigh_joint': 0.1165, 
                'FR_calf_joint': 1.0167,
                'RL_hip_joint': 0.2474,
                'RL_thigh_joint': -2.5143,
                'RL_calf_joint': 1.5880,
                'RR_hip_joint': 0.2927,
                'RR_thigh_joint': -1.4269,
                'RR_calf_joint': 0.8779,
            }
        # init joint angles range
        if init_pose == "upright":
            init_joint_angles_range = {
                'FL_hip_joint': [-0.5786,  0.5786],
                'FL_thigh_joint': [-2.4836,  1.0175],
                'FL_calf_joint': [ 0.6741,  2.3802],
                'FR_hip_joint': [-0.5786,  0.5786], 
                'FR_thigh_joint': [-2.4836,  1.0175],
                'FR_calf_joint': [ 0.6741,  2.3802],
                'RL_hip_joint': [0.1, 0.6],
                'RL_thigh_joint': [-2.65, -2.15],
                'RL_calf_joint': [1.4, 1.7],
                'RR_hip_joint': [-0.6, -0.1],
                'RR_thigh_joint': [-2.65, -2.15],
                'RR_calf_joint': [1.4, 1.7],
            }
        else:
            init_joint_angles_range = {
                key: [value - 0.1, value + 0.1] for (key, value) in init_joint_angles.items()
            }
        randomize_rot = (init_pose == "upright")
    
    class asset(CyberCommonCfg.asset):
        terminate_after_contacts_on = ["base", "head", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot", "RL_thigh", "RR_thigh"] # allow calf, add head
        penalize_contacts_on = ["base", "head", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh"] # stand
        allow_initial_contacts_on = ["foot", "RL_calf", "RR_calf"]
        max_dof_change = 0.3

        #fix_base_link = True
    
    class control(CyberCommonCfg.control):
        stiffness = {'joint': 30.0}
        damping = {'joint': 3.0}
        decimation = 4
        kp_factor_range = [0.8, 1.2]
        kd_factor_range = [0.8, 1.2]
        
    class commands(CyberCommonCfg.commands):
        zero_cmd_threshold = 0.0
        curriculum = use_vel # for no vel expr
        discretize = True
        separate_lin_ang = False
        clip_ang_vel = 0.25 * np.pi
        default_gait_freq = 2.5
        class ranges(CyberCommonCfg.commands.ranges):
            lin_vel_x = [-0.3, 0.3]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-0.5 * np.pi, 0.5 * np.pi]

    class normalization(CyberCommonCfg.normalization):
        class obs_scales(CyberCommonCfg.normalization.obs_scales):
            dof_vel = 0.

    class rewards(CyberCommonCfg.rewards):
        curriculum = (init_pose == "sit")
        cl_init = 0.6
        cl_step = 0.2
        # cl_sigma_terms = ["tracking_pos_sigma"] # TODO
        soft_dof_pos_limit = 0.95
        soft_dof_pos_low = None
        soft_dof_pos_high = None
        soft_torque_limit = 0.5
        
        dof_sigma = 0.1
        tracking_sigma = 0.05#0.05
        tracking_liftup_sigma = 0.03
        # tracking_pos_sigma = 0.003 #changed!!
        tracking_pos_sigma = 0.02
        tracking_ang_sigma = 0.2
        
        liftup_target = 0.42
        # lift_up_threshold = 0.28#0.15 #changed!!!
        lift_up_threshold = [0.15, 0.42]
        scale_factor_low = 0.25
        scale_factor_high = 0.35
        # foot_target = 0.035
        foot_target = 0.05
        if init_pose == "upright":
            allow_contact_steps = 0
        elif init_pose == "sit":
            allow_contact_steps = 30 
        else:
            allow_contact_steps = 50 # !! change with init pose
        before_handtrack_steps = 0 if init_pose == "upright" else 50
        upright_vec = [0.2, 0.0, 1.0]
        ang_rew_mode = "heading"

        class scales(CyberCommonCfg.rewards.scales):
            feet_slip = -0.04 * 10
            feet_clearance_cmd_linear = -300
            collision = -2.0
            torque_limits = -0.01
            tracking_lin_vel = 0.6#0.5*1.0
            tracking_ang_vel = 0.5*0.5
            rear_air = -0.5
            action_rate = -0.03
            action_q_diff = -0.5 * 2 if init_pose == "sit" else 0.
            stand_air = -50 * 0
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            dof_pos_limits = -10
            upright = 1.0
            lift_up_linear = 0.5
            
            foot_twist = -0
            foot_shift = -50
    
    class domain_rand(CyberCommonCfg.domain_rand):
        push_interval_s = 5
        max_push_vel_xy = 0.2
        
        joint_friction_range = [0.03, 0.08]
        joint_damping_range = [0.02, 0.06]
        added_mass_range = [-0.5, 0.5]
        com_displacement_range = [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]]
        
    class terrain(CyberCommonCfg.terrain):
        mesh_type = 'plane' # none, plane, heightfield or trimesh
        curriculum = False

class CyberStandDanceCfgPPO(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "stand_dance_cyber"
        max_iterations = 30000
        save_interval = 300
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

class CyberStandDanceCfgPPOAug(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "stand_dance_cyber_aug"
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPOAugmented'
        max_iterations = 30000
        save_interval = 300
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

class CyberStandDanceCfgPPOEMLP(CyberCommonCfgPPO):
    use_wandb = True
    class runner(CyberCommonCfgPPO.runner):
        experiment_name = "stand_dance_cyber_emlp"
        policy_class_name = 'ActorCriticSymm'
        algorithm_class_name = 'PPO'
        max_iterations = 30000
        save_interval = 300
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'