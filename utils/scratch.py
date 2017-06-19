import os
import numpy as np
import scipy.misc

def create_map():
    src_file_name = '/data/src/monodepth/utils/filenames/kitti_train_files.txt'
    dst_file_name = '/data/src/monodepth/utils/filenames/kitti_train_files_tmp.txt'

    prefix_len = len('2011_10_03/')
    with open(dst_file_name, 'w') as w:
        with open(src_file_name, 'r') as f:
            for line in f:
                l, r = line.split(' ')
                l = l[prefix_len:].replace('.jpg', '.png')
                r = r[prefix_len:].replace('.jpg', '.png')
                w.write('{} {}'.format(l, r))


def create_ply(npy_file, ply_file):
    # Baseline of cameras 02 and 03. See http://www.cvlibs.net/datasets/kitti/setup.php
    baseline_02_03 = 0.54
    # Focal length for 2011_10_03 ride. See respective calib_cam_to_cam.txt.
    focal_length   = 718.856 
    disp  = np.load(npy_file)[0]
    depth = baseline_02_03 * focal_length / (disp.shape[1] * disp)
    
    depth[depth < 1] = 0
    depth[depth > 81] = 82
    depth = depth[:,30:-20]
    count = len(depth[(depth >= 1) & (depth <= 81)])

    scipy.misc.imsave(ply_file + '.depth.png', depth)

    with open(ply_file, 'w') as w:
        w.write(
"""ply
format ascii 1.0
comment created by alexeyk
""")
        w.write('element vertex {}\n'.format(count))
        w.write("""property float x
property float y
property float z
end_header
""")
        for v in range(depth.shape[0]):
            for u in range(depth.shape[1]):
                z = depth[v,u]
                if z >= 1 and z <= 81:
                    w.write('{:.1f} {:.1f} {:.1f}\n'.format(
                            (u - depth.shape[1]/2) * z / focal_length,
                            (v - depth.shape[0]/2) * z / focal_length, z))


npy_file = '/data/tf/log/monodepth/full_06_16_01/disparities.npy'
ply_file = '/data/tf/log/monodepth/full_06_16_01/disparities_pp.ply'
create_ply(npy_file, ply_file)