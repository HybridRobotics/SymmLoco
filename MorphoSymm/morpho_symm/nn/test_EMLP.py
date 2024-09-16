import logging
from typing import Union

import escnn
import numpy as np
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

import escnn
from escnn.nn import FieldType
from hydra import compose, initialize

from morpho_symm.utils.robot_utils import load_symmetric_system

from morpho_symm.nn.EquivariantModules import IsotypicBasis

from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens, Representation, Group

from morpho_symm.nn.EMLP import EMLP


def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three


def get_ground_reaction_forces_rep(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_legs"
    return rep_F

def get_kinematic_three_rep_two(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(2, dtype=int)}
    gens = [permutation_matrix([1, 0])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three

def get_ground_reaction_forces_rep_two(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations['Rd']
    rep_F = {G.identity: np.eye(6, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_front_legs"
    return rep_F

def get_friction_rep(G: Group, rep_kin_three: Representation):
    rep_friction = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(np.kron(np.eye(2,dtype=int), rep_kin_three(g)), np.eye(3,dtype=int))
             for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_friction[h] = rep_h

    rep_friction = group_rep_from_gens(G, rep_friction)
    rep_friction.name = "friction_on_legs"
    return rep_friction

if __name__ == "__main__":
    # Load robot instance and its symmetry group
    initialize(config_path="../cfg/robot", version_base='1.3')
    robot_name = 'a1'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
    robot_cfg = compose(config_name=f"{robot_name}.yaml")
    robot, G = load_symmetric_system(robot_cfg=robot_cfg)

    # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
    gspace = escnn.gspaces.no_base_space(G)
    # Get the relevant group representations.
    rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
    rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
    rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
    rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3


    rep_kin_three = get_kinematic_three_rep_two(G)
    rep = get_ground_reaction_forces_rep_two(G, rep_kin_three)
        
    # nom = torch.tensor([1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4, 1, 0.7, -1.4])
    # print(nom)
    # for g in G.elements[1:]:
    #     sym = torch.from_numpy(rep_TqQJ(g)).float() @ nom.float()
    #     print(sym)

    # raise Exception

    # Define the input and output FieldTypes using the representations of each geometric object.
    in_type = escnn.nn.FieldType(gspace, [rep])
    out_type = escnn.nn.FieldType(gspace, [G.trivial_representation] * 1)
    # Test Invariant EMLP
    emlp = EMLP(in_type, out_type,
                num_hidden_units=128,
                num_layers=3,
                activation="ReLU",
                head_with_activation=False)
    emlp.eval()  # Shut down batch norm
    x = in_type(torch.randn(1, in_type.size))
    y = emlp(x)

    print(x.tensor)
    for g in G.elements:
        g_x = in_type(in_type.transform_fibers(x.tensor, g))  # Compute g · x
        print(g_x.tensor)
        g_y = emlp(g_x)  # Compute g · y
        assert torch.allclose(y.tensor, g_y.tensor, rtol=1e-4, atol=1e-4), \
            f"{g} invariance failed {y.tensor} != {g_y.tensor}"
    print("invariance passed")
