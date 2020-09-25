import numpy as np
from psbody.mesh import Mesh
from opendr.camera import ProjectPoints

def compute_texture_map(source_img, verts, faces, cam, texture_data):
    '''
    Given an image and a mesh aligned with the image (under scale-orthographic projection), project the image onto the
    mesh and return a texture map.
    '''

    x_coords = texture_data.get('x_coords')
    y_coords = texture_data.get('y_coords')
    valid_pixel_ids = texture_data.get('valid_pixel_ids')
    valid_pixel_3d_faces = texture_data.get('valid_pixel_3d_faces')
    valid_pixel_b_coords = texture_data.get('valid_pixel_b_coords')
    img_size = texture_data.get('img_size')

    pixel_3d_points = verts[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                      verts[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                      verts[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]

    vertex_normals = Mesh(verts, faces).estimate_vertex_normals()
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    n_dot_view = pixel_3d_normals[:,2]

    proj_2d_points = ProjectPoints(f=cam[0] * np.ones(2), rt=np.zeros(3), t=np.zeros(3), k=np.zeros(5), c=cam[1:3])
    proj_2d_points.v = pixel_3d_points
    proj_2d_points = np.round(proj_2d_points.r).astype(int)

    texture = np.zeros((img_size, img_size, 3))
    for i, (x, y) in enumerate(proj_2d_points):
        if n_dot_view[i] > 0.0:
            continue
        if x > 0 and x < source_img.shape[1] and y > 0 and y < source_img.shape[0]:
            texture[y_coords[valid_pixel_ids[i]].astype(int), x_coords[valid_pixel_ids[i]].astype(int), :3] = source_img[y, x]
    return texture