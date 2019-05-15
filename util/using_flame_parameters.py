"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.

More information about RingNet is available at https://ringnet.is.tue.mpg.de.
"""
# This function Netralize the pose and expression of the predicted mesh and generates a template mesh with only the identity information
import numpy as np
import chumpy as ch
from smpl_webuser.serialization import load_model
from smpl_webuser.verts import verts_decorated
from psbody.mesh import Mesh


def make_prdicted_mesh_neutral(predicted_params_path, flame_model_path):
    params = np.load(predicted_params_path, allow_pickle=True)
    params = params[()]
    pose = np.zeros(15)
    expression = np.zeros(100)
    shape = np.hstack((params['shape'], np.zeros(300-params['shape'].shape[0])))
    flame_genral_model = load_model(flame_model_path)
    generated_neutral_mesh = verts_decorated(ch.array([0.0,0.0,0.0]),
                        ch.array(pose),
                        ch.array(flame_genral_model.r),
                        flame_genral_model.J_regressor,
                        ch.array(flame_genral_model.weights),
                        flame_genral_model.kintree_table,
                        flame_genral_model.bs_style,
                        flame_genral_model.f,
                        bs_type=flame_genral_model.bs_type,
                        posedirs=ch.array(flame_genral_model.posedirs),
                        betas=ch.array(np.hstack((shape,expression))),#betas=ch.array(np.concatenate((theta[0,75:85], np.zeros(390)))), #
                        shapedirs=ch.array(flame_genral_model.shapedirs),
                        want_Jtr=True)
    neutral_mesh = Mesh(v=generated_neutral_mesh.r, f=generated_neutral_mesh.f)
    return neutral_mesh
