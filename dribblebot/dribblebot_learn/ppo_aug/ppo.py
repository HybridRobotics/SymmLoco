import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from dribblebot_learn.ppo_aug import ActorCritic
from dribblebot_learn.ppo_aug import RolloutStorage
from dribblebot_learn.ppo_aug import caches

from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two

import escnn
from escnn.nn import FieldType
from hydra import compose, initialize

from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.robot_utils import load_symmetric_system


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 5.e-5  # 5.e-4
    adaptation_module_learning_rate = 5.e-5
    num_adaptation_module_substeps = 1
    schedule = 'fixed'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        
        PPO_Args.adaptation_labels = self.actor_critic.adaptation_labels
        PPO_Args.adaptation_dims = self.actor_critic.adaptation_dims
        PPO_Args.adaptation_weights = self.actor_critic.adaptation_weights
        
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate

        global G
        initialize(config_path="../../../MorphoSymm/morpho_symm/cfg/robot", version_base='1.3')
        robot_name = 'a1'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
        robot_cfg = compose(config_name=f"{robot_name}.yaml")
        robot, G = load_symmetric_system(robot_cfg=robot_cfg)

        # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
        gspace = escnn.gspaces.no_base_space(G)
        # Get the relevant group representations.
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
        rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
        rep_kin_three = get_kinematic_three_rep_two(G)
        # Define the input and output FieldTypes using the representations of each geometric object.
        # Representation of x := [q, v] ∈ Q_js x TqQ_js      =>    ρ_X_js(g) := ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
        obs_transition = [rep_O3, rep_O3, rep_O3, rep_O3_pseudo] + [G.trivial_representation] * 12 + [rep_QJ] * 4 + [rep_kin_three] * 3
        priv_obs_transition = [rep_O3, rep_O3]
        obs_history_transition = obs_transition * 15
        self.obs_field_type = FieldType(gspace, obs_transition)
        # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
        self.action_field_type = FieldType(gspace, [rep_QJ])

        self.priv_obs_field_type = FieldType(gspace, priv_obs_transition)
        self.obs_history_field_type = FieldType(gspace, obs_history_transition)

        self.num_replica = len(G.elements)
        self.G = G

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs * self.num_replica, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.augment_transitions()
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def augment_transitions(self):
        t = self.transition
        G = self.G
        t.observations = torch.cat([t.observations] + [self.obs_field_type.transform_fibers(t.observations, g) for g in G.elements[1:]], dim=0)
        t.privileged_observations = torch.cat([t.privileged_observations] + [self.priv_obs_field_type.transform_fibers(t.privileged_observations, g) for g in G.elements[1:]], dim=0)
        t.observation_histories = torch.cat([t.observation_histories] + [self.obs_history_field_type.transform_fibers(t.observation_histories, g) for g in G.elements[1:]], dim=0)
        t.actions = torch.cat([t.actions] + [self.action_field_type.transform_fibers(t.actions, g) for g in G.elements[1:]], dim=0)
        t.rewards = torch.cat([t.rewards] * self.num_replica, dim=0)
        t.dones = torch.cat([t.dones] * self.num_replica, dim=0)
        t.values = torch.cat([t.values] * self.num_replica, dim=0)
        t.actions_log_prob = torch.cat([t.actions_log_prob] * self.num_replica, dim=0)
        t.action_mean = torch.cat([t.action_mean] + [self.action_field_type.transform_fibers(t.action_mean, g) for g in G.elements[1:]], dim=0)
        t.action_sigma = torch.abs(torch.cat([t.action_sigma] + [self.action_field_type.transform_fibers(t.action_sigma, g) for g in G.elements[1:]], dim=0))
        t.env_bins = torch.cat([t.env_bins] * self.num_replica, dim=0)

    def augment_values(self, values):
        values = torch.cat([values] * self.num_replica, dim=0)
        return values

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        last_values = self.augment_values(last_values)
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        
        mean_adaptation_losses = {}
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0
        
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_history_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step, only update concurrent state estimation module, not policy network
            if len(PPO_Args.adaptation_labels) > 0:

                for epoch in range(PPO_Args.num_adaptation_module_substeps):

                    adaptation_pred = self.actor_critic.get_student_latent(obs_history_batch)
                    with torch.no_grad():
                        adaptation_target = privileged_obs_batch
                    adaptation_loss = 0
                    for idx, (label, length, weight) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims, PPO_Args.adaptation_weights)):

                        start, end = label_start_end[label]
                        selection_indices = torch.linspace(start, end - 1, steps=end - start, dtype=torch.long)

                        idx_adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices] * weight,
                                                        adaptation_target[:, selection_indices] * weight)
                        mean_adaptation_losses[label] += idx_adaptation_loss.item()

                        adaptation_loss += idx_adaptation_loss

                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()

                    mean_adaptation_module_loss += adaptation_loss.item()
                    mean_adaptation_module_test_loss += 0  # adaptation_test_loss.item()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        for label in PPO_Args.adaptation_labels:
            mean_adaptation_losses[label] /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses
