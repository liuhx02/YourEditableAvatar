import numpy as np
import torch
import open3d as o3d
import json

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from 3D Gaussian Splatting (which copied from Plenoxels)

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def trans_gs_mesh(mesh, metadata_path, R_path):
    R = np.load(R_path)
    R_inv = np.linalg.inv(R)
    verts = np.asarray(mesh.vertices)
    verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts = np.dot(R_inv, verts.T).T
    
    sdfstudio_to_colmap = np.array([
        [-0.,  1.,  0.,  0.],
        [ 1.,  0., -0., -0.],
        [-0., -0., -1.,  0.],
        [ 0.,  0.,  0.,  1.]]
    )
    metadata = json.load(open(metadata_path))
    worldtogt = np.array(metadata['worldtogt'])
    verts = np.dot(verts, worldtogt.T)
    verts = np.dot(verts, sdfstudio_to_colmap)
    verts = verts[:, :3]
    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh

def transfer_pcd_color(sparse_ply_path, dense_mesh, k=20):
    sparse_pcd = o3d.io.read_point_cloud(sparse_ply_path)
    sparse_points = np.asarray(sparse_pcd.points)
    sparse_colors = np.asarray(sparse_pcd.colors)
    dense_points = np.asarray(dense_mesh.vertices)
    # filter out white points on the background
    threshold = 0.95
    non_white_mask = ~np.all(sparse_colors > threshold, axis=1)
    sparse_points = sparse_points[non_white_mask]
    sparse_colors = sparse_colors[non_white_mask]
    
    # knn
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(sparse_colors)
    sparse_tree = o3d.geometry.KDTreeFlann(filtered_pcd)
    dense_colors = np.zeros((len(dense_points), 3))
    for i, point in enumerate(dense_points):
        [k, idx, _] = sparse_tree.search_knn_vector_3d(point, k)
        if k == 1:
            dense_colors[i] = sparse_colors[idx[0]]
        else:
            dense_colors[i] = np.mean(sparse_colors[idx], axis=0)

    colored_dense_mesh = o3d.geometry.TriangleMesh()
    colored_dense_mesh.vertices = o3d.utility.Vector3dVector(dense_points)
    colored_dense_mesh.triangles = dense_mesh.triangles
    colored_dense_mesh.vertex_colors = o3d.utility.Vector3dVector(dense_colors)
    
    return colored_dense_mesh