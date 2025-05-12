from gettext import dpgettext
#from util_io import read_pfm, write_pfm
import numpy as np
from scipy.interpolate import interp1d
import imageio
import trimesh
import cv2
from icecream import ic

import os


import torch

if torch.cuda.is_available():
    #print('cuda available')
    os.environ["PYOPENGL_PLATFORM"] = "egl"  # gpu rendering
else:
    # https://pyrender.readthedocs.io/en/latest/install/index.html#installing-osmesa
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # cpu-rendering

ic(os.environ["PYOPENGL_PLATFORM"])
import pyglet

pyglet.options["shadow_window"] = False

import pyrender
import time


def resize_rgb(rgb, trgt_H, trgt_W):
    rgb = cv2.resize(rgb, (trgt_W, trgt_H), interpolation=cv2.INTER_AREA)
    return rgb


def resize_depth(depth, trgt_H, trgt_W):
    depth = cv2.resize(depth, (trgt_W, trgt_H), interpolation=cv2.INTER_NEAREST)
    return depth


def generate_faces(H, W):
    vert1 = np.arange(H * W).reshape((H, W))[: H - 1, : W - 1]
    ll_vert2 = vert1 + W
    ll_vert3 = ll_vert2 + 1

    ur_vert2 = ll_vert3
    ur_vert3 = vert1 + 1

    ll_faces = np.stack([vert1, ll_vert2, ll_vert3], axis=-1).reshape((-1, 3))
    ur_faces = np.stack([vert1, ur_vert2, ur_vert3], axis=-1).reshape((-1, 3))
    return np.concatenate([ll_faces, ur_faces], axis=0).astype(int)


def depth2pcd(depth, K, C2W):
    H, W = depth.shape[:2]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pts = np.stack((u, v, np.ones_like(u)), axis=-1)
    pts = (pts.reshape((-1, 3)) @ np.linalg.inv(K[:3, :3]).T).reshape((H, W, 3)) * depth[:, :, np.newaxis]
    vertices = pts.reshape((-1, 3))
    vertices = np.concatenate([vertices, np.ones_like(vertices[:, 0:1])], axis=-1) @ C2W.T
    vertices = vertices[:, :3].reshape((H, W, 3))
    return vertices.astype(np.float32)


def unproject_depth(
    out_ply, rgb, depth, K, C2W, scale_factor=1.0, add_faces=False, prune_edge_faces=True, prune_angle=70.0
):
    H, W = rgb.shape[:2]
    K = np.copy(K)
    # K[0, :3] *= W
    # K[1, :3] *= H
    if not np.isclose(scale_factor, 1.0):
        rgb = resize_rgb(rgb, int(H * scale_factor), int(W * scale_factor))
        depth = resize_depth(depth, int(H * scale_factor), int(W * scale_factor))
        K[0, :3] *= rgb.shape[1] / W
        K[1, :3] *= rgb.shape[0] / H
        H, W = rgb.shape[:2]

    vertices = depth2pcd(depth, K, C2W=np.eye(4).astype(np.float32)).reshape((-1, 3))
    vertex_colors = np.clip(rgb * 255.0, 0.0, 255.0).reshape((-1, 3)).astype(np.uint8)

    if not add_faces:
        mask = (depth > 0.0).reshape((-1,))
        geometry = trimesh.PointCloud(vertices=vertices[mask], colors=vertex_colors[mask])
    else:
        faces = generate_faces(H, W)

        if prune_edge_faces:
            face_verts_1, face_verts_2, face_verts_3 = (
                vertices[faces[:, 0]],
                vertices[faces[:, 1]],
                vertices[faces[:, 2]],
            )
            face_center = (face_verts_1 + face_verts_2 + face_verts_3) / 3.0
            face_normals = np.cross(face_verts_2 - face_verts_1, face_verts_3 - face_verts_2, axis=-1)
            face_normals = face_normals / (np.linalg.norm(face_normals, axis=-1, keepdims=True)+1e-8)
            # assume camera center is (0, 0, 0)
            view_dirs = -face_center / (np.linalg.norm(face_center, axis=-1, keepdims=True)+1e-8)
            valid_mask = np.sum(face_normals * view_dirs, axis=-1) > np.cos(np.deg2rad(prune_angle))

            # ic(faces.shape, valid_mask.shape, face_normals.shape, view_dirs.shape, face_center.shape)
            faces = faces[valid_mask]
            # TODO: remove vertices that are not used by any face

        geometry = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        # vertex_mask = (depth > 1e-3).reshape((-1,))
        vertex_mask = geometry.vertices[:, 2] > 0.0
        # ic(depth.shape, vertices.shape, vertex_mask.shape)
        geometry.update_vertices(vertex_mask)

        geometry.remove_unreferenced_vertices()

    # transform vertices to world frame
    geometry.apply_transform(C2W.astype(np.float32))

    if out_ply is not None:
        geometry.export(out_ply)
    return geometry


def flip_yz(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def render_view(H, W, K, C2W, mesh, ref_C2W=None):
    # tic = time.time()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=np.array([0, 0, 0]))
    scene.add(mesh)
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    # https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=200)
    cam_node = scene.add(cam, pose=flip_yz(C2W))
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    r.delete()
    scene.remove_node(cam_node)  # remove camera from the scene
    color = color.astype(np.float32) / 255.0
    if ref_C2W is not None:  # change to reference view depth
        mask = depth > 0.0
        pcd = depth2pcd(depth, K, np.linalg.inv(ref_C2W) @ C2W)
        depth = pcd[:, :, 2] * mask
        # ic(depth.min(), depth.max())
    depth = depth.astype(np.float32)
    # ic("render_view", time.time() - tic)
    return color, depth


def render_multiviews(H, W, K, C2Ws, meshes, ref_C2W=None):
    # tic = time.time()
    scene = pyrender.Scene(bg_color=np.array([0, 0, 0]))
    if not isinstance(meshes, list):
        meshes = [
            meshes,
        ]
    for mesh in meshes:
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    # https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.001, zfar=1e8)
    #print("zfar 1e8")
    colors, depths, colors_behind, depths_behind = [], [], [], []
    for C2W in C2Ws:
        cam_node = scene.add(cam, pose=flip_yz(C2W))
        # color, depth, color_behind, depth_behind = r.render(
        #     scene, flags=pyrender.constants.RenderFlags.FLAT  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
        # )
        color, depth = r.render(
            scene, flags=pyrender.constants.RenderFlags.FLAT  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
        )
        
        #occlude_mask = depth_behind > depth

        # ic(np.min(color), np.max(color), np.min(depth), np.max(depth))
        scene.remove_node(cam_node)  # remove camera from the scene
        color = color.astype(np.float32) / 255.0
        colors.append(color)

        # color_behind = color_behind.astype(np.float32) / 255.0
        # color_behind = color_behind * occlude_mask[:, :, np.newaxis]
        # depth_behind = depth_behind * occlude_mask

        # colors_behind.append(color_behind)

        if ref_C2W is not None:  # change to reference view depth
            mask = depth > 0.0
            pcd = depth2pcd(depth, K, np.linalg.inv(ref_C2W) @ C2W)
            depth = pcd[:, :, 2] * mask
            # ic(depth.min(), depth.max())
        depths.append(depth.astype(np.float32))
        #depths_behind.append(depth_behind.astype(np.float32))
    r.delete()
    colors = np.stack(colors, axis=0)
    depths = np.stack(depths, axis=0)
    #colors_behind = np.stack(colors_behind, axis=0)
    #depths_behind = np.stack(depths_behind, axis=0)
    # ic("render_view", time.time() - tic)
    return colors, depths #, colors_behind, depths_behind


def load_depth(disp_fi, disp_rescale=3.0, h=None, w=None):
    """
    https://github.com/vt-vl-lab/3d-photo-inpainting/blob/de0446740a3726f3de76c32e78b43bd985d987f9/main.py#L73
    https://github.com/vt-vl-lab/3d-photo-inpainting/blob/de0446740a3726f3de76c32e78b43bd985d987f9/utils.py#L942
    """
    disp = read_pfm(disp_fi)

    disp = disp - disp.min()
    # disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
    disp = (disp / disp.max()) * disp_rescale
    if h is not None and w is not None:
        disp = resize_depth(disp, h, w)
    depth = 1.0 / np.maximum(disp, 0.05)

    return depth


def load_image(img_fpath, scale_factor=1, ensure_multiple_of=1, ensure_min_size=-1):
    img = imageio.imread(img_fpath).astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    img = img[..., :3]

    H, W = img.shape[:2]
    new_H, new_W = H, W
    if scale_factor != 1:
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    if ensure_min_size > 0 and min([new_H, new_W]) < ensure_min_size:
        if H > W:
            new_W = ensure_min_size
            new_H = int(H / W * ensure_min_size)
        else:
            new_H = ensure_min_size
            new_W = int(W / H * ensure_min_size)

    new_H, new_W = (
        ((new_H - 1) // ensure_multiple_of + 1) * ensure_multiple_of,
        ((new_W - 1) // ensure_multiple_of + 1) * ensure_multiple_of,
    )
    if new_H != H or new_W != W:
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return img


def guess_image_K(image):
    H, W = image.shape[:2]
    K = np.array(
        [[max(H, W), 0.0, W / 2.0, 0.0], [0.0, max(H, W), H / 2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ).astype(np.float32)
    return K


def guess_K(H, W):
    K = np.array(
        [[max(H, W), 0.0, W / 2.0, 0.0], [0.0, max(H, W), H / 2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ).astype(np.float32)
    return K


def load_image_and_K(img_fpath, scale_factor=1, ensure_multiple_of=1, ensure_min_size=-1):
    """
    https://github.com/vt-vl-lab/3d-photo-inpainting/blob/60ce4fcc5f8dc37a2b65bb72ef6287addc024bbf/utils.py#L874
    """
    img = load_image(img_fpath, scale_factor, ensure_multiple_of, ensure_min_size)

    H, W = img.shape[:2]
    K = np.array(
        [[max(H, W), 0.0, W / 2.0, 0.0], [0.0, max(H, W), H / 2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ).astype(np.float32)
    return img, K


def FOV2K(fov, W, H, is_hfov=True):
    if is_hfov:
        f = W / 2.0 / np.tan(np.deg2rad(fov / 2.0))
    else:
        f = H / 2.0 / np.tan(np.deg2rad(fov / 2.0))
    K = np.eye(4)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K.astype(np.float32)


def path_planning(num_frames, x, y, z, path_type=""):
    if path_type == "straight-line":
        corner_points = np.array([[0, 0, 0], [(0 + x) * 0.5, (0 + y) * 0.5, (0 + z) * 0.5], [x, y, z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, num_frames)
        cs = interp1d(corner_t, corner_points, axis=0, kind="quadratic")
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
    elif path_type == "double-straight-line":
        corner_points = np.array([[-x, -y, -z], [0, 0, 0], [x, y, z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, num_frames)
        cs = interp1d(corner_t, corner_points, axis=0, kind="quadratic")
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
    elif path_type == "circle":
        xs, ys, zs = [], [], []
        for frame_id, bs_shift_val in enumerate(np.arange(-2.0, 2.0, (4.0 / num_frames))):
            xs += [np.cos(bs_shift_val * np.pi) * 1 * x]
            ys += [np.sin(bs_shift_val * np.pi) * 1 * y]
            zs += [np.cos(bs_shift_val * np.pi / 2.0) * 1 * z]
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    return xs, ys, zs


def get_render_poses():
    """
    https://github.com/vt-vl-lab/3d-photo-inpainting/blob/60ce4fcc5f8dc37a2b65bb72ef6287addc024bbf/utils.py#L839
    """
    config = {
        "traj_types": ["double-straight-line", "double-straight-line", "circle", "circle", "circle"],
        "x_shift_range": [0.00, 0.00, -0.02, -0.02, -0.04],
        "y_shift_range": [0.00, 0.00, -0.02, -0.00, -0.04],
        "z_shift_range": [-0.05, -0.05, -0.07, -0.07, -0.04],
        "video_postfix": ["dolly-zoom-in", "zoom-in", "circle", "swing", "circle"],
        "num_frames": 240,
    }
    assert (
        len(config["traj_types"])
        == len(config["x_shift_range"])
        == len(config["y_shift_range"])
        == len(config["z_shift_range"])
        == len(config["video_postfix"])
    ), "The number of elements in 'traj_types', 'x_shift_range', 'y_shift_range', 'z_shift_range' and \
                'video_postfix' should be equal."
    tgts_poses = []
    for traj_idx in range(len(config["traj_types"])):
        tgt_poses = []
        sx, sy, sz = path_planning(
            config["num_frames"],
            config["x_shift_range"][traj_idx],
            config["y_shift_range"][traj_idx],
            config["z_shift_range"][traj_idx],
            path_type=config["traj_types"][traj_idx],
        )
        for xx, yy, zz in zip(sx, sy, sz):
            c2w = np.eye(4)
            c2w[:3, -1] = np.array([xx, yy, zz])
            tgt_poses.append(c2w)
        tgt_poses = np.stack(tgt_poses, axis=0).astype(np.float32)
        tgts_poses.append(tgt_poses)
    return tgts_poses


def get_training_poses():
    return get_render_poses()[-1]
    # return get_render_poses()[-3]


def get_shift_poses(shift_amount=0.06):
    tgts_poses = []

    tmp_c2w = np.eye(4).astype(np.float32)
    tmp_c2w[0, 3] = shift_amount
    tgts_poses.append(tmp_c2w)

    tmp_c2w = np.eye(4).astype(np.float32)
    tmp_c2w[0, 3] = -shift_amount
    tgts_poses.append(tmp_c2w)

    tmp_c2w = np.eye(4).astype(np.float32)
    tmp_c2w[1, 3] = shift_amount
    tgts_poses.append(tmp_c2w)

    tmp_c2w = np.eye(4).astype(np.float32)
    tmp_c2w[1, 3] = -shift_amount
    tgts_poses.append(tmp_c2w)

    return np.stack(tgts_poses, axis=0)


def overlay_mask(rgb, mask, color=[0.1, 0.8, 0.1], mask_opacity=0.35):
    color = np.ones_like(rgb) * np.array(color).reshape((1, 1, 3))
    alpha = (mask_opacity * mask)[:, :, np.newaxis]
    rgb = color * alpha + rgb * (1.0 - alpha)
    return rgb


def mesh_warping(fname, fname2, aligned_depth, imgname_dict1, imgname_dict2, target_res=256, n=None):

    # refimg, ratio1 = resize_with_padding(fname, target_res, return_unpadded=True) # resize with larger side target_res, no padding yet
    # tarimg, ratio2 = resize_with_padding(fname2, target_res, return_unpadded=True) 
    # aligned_depth = cv2.resize(aligned_depth, dsize=(refimg.shape[1], refimg.shape[0]), interpolation=cv2.INTER_AREA) # dsize is WxH


    refimg = np.array(fname)
    tarimg = np.array(fname2)

    # fx1, fy1 = fxfy_from_intrinsics(imgname_dict1['intrinsics'])
    # fx2, fy2 = fxfy_from_intrinsics(imgname_dict2['intrinsics'])
    # fx1, fy1 = fx1 * ratio1, fy1 * ratio1  # scale by resized factor
    # fx2, fy2 = fx2 * ratio2, fy2 * ratio2 

    K1 = intrinsics_to_matrix(imgname_dict1['intrinsics'])
    K2 = intrinsics_to_matrix(imgname_dict2['intrinsics'])
    # K1[0, 0], K1[1, 1] = fx1, fy1
    # K1[0, 2], K1[1, 2] = refimg.shape[1]/2, refimg.shape[0]/2 # cx,cy is w/2,h/2, respectively
    # K2[0, 0], K2[1, 1] = fx2, fy2
    # K2[0, 2], K2[1, 2] = tarimg.shape[1]/2, tarimg.shape[0]/2

    original_pose = np.linalg.inv(extrinsics_to_matrix(imgname_dict1['extrinsics']))
    new_pose = np.linalg.inv(extrinsics_to_matrix(imgname_dict2['extrinsics']))

    outply = os.path.join("testmesh", n.split('/')[-1][:-4]+'.ply')
    mesh_hole = unproject_depth(outply, refimg/255., aligned_depth, K1, original_pose, scale_factor=1.0, add_faces=True, prune_edge_faces=True, prune_angle=70)
    outh, outw = tarimg.shape[:2]
    rgb_nohole, _ = render_view(outh, outw, K2, new_pose, mesh_hole)

    return (rgb_nohole*255).astype(np.uint8)

def imgnames_to_warpname(imgname1, imgname2):
    # splitting filenames into directories: warped_depth_imgs/category/commons/subcat1/imgname1/to/subcat2/imgname2.png
    cat = imgname1.split('main/images/')[-1].split('/commons')[0]
    subcat = imgname1.split('commons/')[-1].split('/0/pictures')[0]
    fname = imgname1.split('/')[-1][:-4]
    subcat2 = imgname2.split('commons/')[-1].split('/0/pictures')[0]
    fname2 = imgname2.split('/')[-1][:-4]
    warpname = os.path.join( cat, 'commons', subcat, "imgname:", fname, 'to', subcat2, "imgname:", fname2 )
    warpname = warpname.replace('/', '_')
    hash_object = hashlib.sha1(warpname.encode())
    hex_dig = hash_object.hexdigest() 
    return hex_dig

def imgname_to_depthname(aligned_depth_path, imgname):
    # e.g. '/share/phoenix/nfs05/S8/jt664/WikiSFM/data/main/images/Boekentoren_(Ghent)/commons/Boekentoren_(Ghent)/0/pictures/Gent Rozier Boekentoren-PM 35307.jpg'
    # --> 'aligned_depth_path/Boekentoren_(Ghent)/commons/Boekentoren_(Ghent)/aligned_depth_maps/Gent Rozier Boekentoren-PM 35307.npy'
    fname = imgname.split('main/images/')[-1][:-4]+'.npy'
    fname = fname.replace('0/pictures/', 'aligned_depth_maps/')
    os.makedirs(os.path.join(aligned_depth_path, os.path.dirname(fname)), exist_ok=True)
    return os.path.join(aligned_depth_path, fname)

if __name__ == "__main__":
    import os, sys, glob, pickle
    from PIL import Image
    import numpy as np

    submodule_path = ("/sensei-fs-3/users/gchou/video-depth/data")
    assert os.path.exists(submodule_path)
    sys.path.insert(0, submodule_path)
    from sintel_io import depth_read, cam_read
    
    # try warping sintel 
    # all poses here are c2w, so if the warping works, we know if given poses are c2w or w2c

    img = Image.open('/mnt/localssd/sintel/images/training/clean/temple_2/frame_0001.png').convert('RGB')
    img = np.array(img) / 255.0
    h, w = img.shape[:2]
    depthmap = depth_read('/mnt/localssd/sintel/depth/training/depth/temple_2/frame_0001.dpt')
    intrinsics, w2c_original_pose = cam_read('/mnt/localssd/sintel/depth/training/camdata_left/temple_2/frame_0001.cam')
    w2c_original_pose = np.vstack((w2c_original_pose, np.array([0, 0, 0, 1])))
    c2w_original_pose = np.linalg.inv(w2c_original_pose)
    _, w2c_target_pose = cam_read('/mnt/localssd/sintel/depth/training/camdata_left/temple_2/frame_0050.cam')
    w2c_target_pose = np.vstack((w2c_target_pose, np.array([0, 0, 0, 1])))
    c2w_target_pose = np.linalg.inv(w2c_target_pose)
    
    mesh = unproject_depth('test.ply', img, depthmap, intrinsics, c2w_original_pose, scale_factor=1.0, add_faces=True, prune_edge_faces=True)
    warped_image, _ = render_view(h, w, intrinsics, c2w_target_pose, mesh)

    Image.fromarray( (warped_image*255).astype(np.uint8) ).save('test.png')
