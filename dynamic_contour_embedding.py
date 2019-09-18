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
All rights reserved.
More information about RingNet is available at https://ringnet.is.tue.mpg.de.
"""

## A function to load the dynamic contour on a template mesh
## Please cite the updated citaion from https://ringnet.is.tue.mpg.de if you use the dynamic contour for FLAME

import numpy as np
import pyrender
import trimesh
from smpl_webuser.serialization import load_model
from psbody.mesh import Mesh

def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    # function: evaluation 3d points given mesh and landmark embedding
    # modified from https://github.com/Rubikplayer/flame-fitting/blob/master/fitting/landmarks.py
    dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

def load_dynamic_contour(template_flame_path='None', contour_embeddings_path='None', angle=0):
    template_mesh = Mesh(filename=template_flame_path)
    contour_embeddings_path = contour_embeddings_path
    dynamic_lmks_embeddings = np.load(contour_embeddings_path, allow_pickle=True).item()
    lmk_face_idx = dynamic_lmks_embeddings['lmk_face_idx'][angle]
    lmk_b_coords = dynamic_lmks_embeddings['lmk_b_coords'][angle]
    dynamic_lmks = mesh_points_by_barycentric_coordinates(template_mesh.v, template_mesh.f, lmk_face_idx, lmk_b_coords)

    # Visualization of the pose dependent contour on the template mesh
    vertex_colors = np.ones([template_mesh.v.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(template_mesh.v, template_mesh.f,
                               vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(dynamic_lmks), 1, 1))
    tfs[:, :3, 3] = dynamic_lmks
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    # angle = 35.0 #in degrees
    angle = 0.0 #in degrees
    # angle = -16.0 #in degrees
    contour_embeddings_path = './flame_model/flame_dynamic_embedding.npy'
    load_dynamic_contour(template_flame_path='./flame_model/FLAME_sample.ply', contour_embeddings_path=contour_embeddings_path, angle=int(angle))
