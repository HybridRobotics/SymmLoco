import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal

import numpy as np
import escnn
from escnn.nn import FieldType, EquivariantModule, GeometricTensor
from hydra import compose, initialize

from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]
    
    adaptation_labels = []
    adaptation_dims = []
    adaptation_weights = []

    use_decoder = False


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()
        
        self.adaptation_labels = AC_Args.adaptation_labels
        self.adaptation_dims = AC_Args.adaptation_dims
        self.adaptation_weights = AC_Args.adaptation_weights

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        global G
        # Load robot instance and its symmetry group
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
        obs_transition = ([rep_O3, rep_O3, rep_O3, rep_O3_pseudo] + [G.trivial_representation] * 12 + [rep_QJ] * 4 + [rep_kin_three] * 3) * 15
        priv_obs_transition = [rep_O3, rep_O3]

        adaptation_in_field_type = FieldType(gspace, obs_transition)
        adaptation_out_field_type = FieldType(gspace, priv_obs_transition)
        actor_in_field_type = FieldType(gspace, obs_transition + priv_obs_transition)
        actor_out_field_type = FieldType(gspace, [rep_QJ])
        critic_in_field_type = FieldType(gspace, obs_transition + priv_obs_transition)
        critic_out_field_type = FieldType(gspace, [G.trivial_representation])

        self.gspace = gspace
        self.adaptation_in_field_type = adaptation_in_field_type
        self.actor_in_field_type = actor_in_field_type
        self.critic_in_field_type = critic_in_field_type

        self.actor_body = SimpleEMLP(actor_in_field_type, actor_out_field_type,
            hidden_dims = AC_Args.actor_hidden_dims,
            activation = AC_Args.activation)

        self.adaptation_module = SimpleEMLP(adaptation_in_field_type, adaptation_out_field_type,
            hidden_dims = AC_Args.adaptation_module_branch_hidden_dims,
            activation = AC_Args.activation)
        
        self.critic_body = SimpleEMLP(critic_in_field_type, critic_out_field_type,
            hidden_dims = AC_Args.critic_hidden_dims,
            activation = AC_Args.activation)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        model_parameters = filter(lambda p: p.requires_grad, self.adaptation_module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Adaptation Module #Params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, self.actor_body.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Actor #Params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, self.critic_body.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Critic #Params: ", params)

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history):
        observation_history_tran = self.adaptation_in_field_type(observation_history)
        latent_tran = self.adaptation_module(observation_history_tran)
        latent = latent_tran.tensor
        mean = self.actor_body(self.actor_in_field_type(torch.cat((observation_history, latent), dim=-1))).tensor
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        observation_history_tran = self.adaptation_in_field_type(observation_history)
        latent_tran = self.adaptation_module(observation_history_tran)
        latent = latent_tran.tensor
        actions_mean = self.actor_body(self.actor_in_field_type(torch.cat((observation_history, latent), dim=-1))).tensor
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(self.actor_in_field_type(torch.cat((observation_history, privileged_info), dim=-1))).tensor
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        value = self.critic_body(self.critic_in_field_type(torch.cat((observation_history, privileged_observations), dim=-1))).tensor
        return value

    def get_student_latent(self, observation_history):
        observation_history_tran = self.adaptation_in_field_type(observation_history)
        return self.adaptation_module(observation_history_tran).tensor


class SimpleEMLP(EquivariantModule):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 hidden_dims = [256, 256, 256],
                 bias: bool = True,
                 activation: str = "ReLU"):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup
        
        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(gspace, [group.regular_representation] * int((hidden_dims[n] / group.order())))

            self.net.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        self.net.add_module(f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                             escnn.nn.Linear(layer_in_type, out_type, bias=bias))

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.net(x)

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        sequential = nn.Sequential()
        for name, module in self.net.named_children():
            sequential.add_module(name, module.export())
        return sequential