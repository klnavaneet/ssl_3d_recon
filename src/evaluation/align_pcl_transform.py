###############################################################################
# Find ideal global transformation for point cloud allignment between
# prediction and ground truth pcl
# Perform the optimization on validation set and use the same parameters for
# test set - Need to first save the val point clouds using save_pcl.py
###############################################################################

import os, sys
from os.path import join
import glob
import random
import pdb

import numpy as np
import tensorflow as tf

sys.path.append('../')
sys.path.append('../chamfer_utils/')
sys.path.append('../utils/')
from get_losses import get_3d_loss
from shapenet_taxonomy import shapenet_category_to_id


# Change category id as required
categ = 'car'
categ = shapenet_category_to_id[categ]
N_PTS = 1024
N_ITERS = 5000
batch_size = 1
lr = 1e-3
print_n = 100
save_outputs = True

# Change experiment directory as required
exp_dir = '../../expts/temp'
pred_pcl_dir = join(exp_dir, 'log_proj_pcl_val')
gt_pcl_dir = '../../data/ShapeNet_v1/%s'%categ
out_dir = join(exp_dir, 'log_proj_pcl_test_rot')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

models_list = sorted(glob.glob(join(pred_pcl_dir, '*.npy')))
N_MODELS = len(models_list)
random.shuffle(models_list)

def get_feed_dict(cnt, models_list):
    pred_pcl_list = []; gt_pcl_list = [];
    for i in range(batch_size):
        idx = (cnt+i)%len(models_list)
        model_name = models_list[idx].split('/')[-1].split('_')[1]
        _pred_pcl = np.load(models_list[idx])[:,:3]
        _gt_pcl = np.load(join(gt_pcl_dir, model_name,
            'pcl_1024_fps_trimesh.npy'))
        gt_pcl_list.append(_gt_pcl)
        pred_pcl_list.append(_pred_pcl)
    pred_pcl_list = np.stack(pred_pcl_list, 0)
    gt_pcl_list = np.stack(gt_pcl_list, 0)
    feed_dict = {gt_pcl: gt_pcl_list, pred_pcl: pred_pcl_list}
    return feed_dict


def get_rotmat(angles, use_tilt=True):
    angles = tf.tile(angles, [batch_size])
    az, el, tilt = tf.split(angles, 3, -1)
    rotmat_az=[
                    [tf.ones_like(az),tf.zeros_like(az),tf.zeros_like(az)],
                    [tf.zeros_like(az),tf.cos(az),-tf.sin(az)],
                    [tf.zeros_like(az),tf.sin(az),tf.cos(az)]
                    ]

    rotmat_el=[
                    [tf.cos(el),tf.zeros_like(az), tf.sin(el)],
                    [tf.zeros_like(az),tf.ones_like(az),tf.zeros_like(az)],
                    [-tf.sin(el),tf.zeros_like(az), tf.cos(el)]
                    ]
    rotmat_tilt =[
                    [tf.cos(tilt),-tf.sin(tilt),tf.zeros_like(az)],
                    [tf.sin(tilt), tf.cos(tilt),tf.zeros_like(az)],
                    [tf.zeros_like(az),tf.zeros_like(az),tf.ones_like(az)]
                    ]

    rotmat_az = tf.transpose(tf.stack(rotmat_az, 0), [2,0,1])
    rotmat_el = tf.transpose(tf.stack(rotmat_el, 0), [2,0,1])
    rotmat_tilt = tf.transpose(tf.stack(rotmat_tilt, 0), [2,0,1])

    if use_tilt:
        rotmat = tf.matmul(rotmat_tilt, tf.matmul(rotmat_el, rotmat_az))
    else:
        rotmat = tf.matmul(rotmat_el, rotmat_az)
    return rotmat


pred_pcl = tf.placeholder(tf.float32, (batch_size, N_PTS, 3),
        'predicted_pcl')
gt_pcl = tf.placeholder(tf.float32, (batch_size, N_PTS, 3),
        'gt_pcl')
angles = tf.Variable([29.*np.pi/180., 0.*np.pi/180., 270.*np.pi/180.],
                      tf.float32, name='transformation_matrix')
angles_gt = tf.constant([np.pi/3., np.pi/6., np.pi/2.], tf.float32,
                         name='gt_transformation_matrix')

# Verify code by rotating gt by known angle and predict the same
rotmat_gt = get_rotmat(angles_gt, use_tilt=False)
gt_pcl_rot = tf.transpose(tf.matmul(
                            rotmat_gt, tf.transpose(gt_pcl, [0,2,1])), [0,2,1])
rot_mat = get_rotmat(angles)

pred_rot = tf.transpose(tf.matmul(
                            rot_mat, tf.transpose(pred_pcl, [0,2,1])), [0,2,1])
loss = get_3d_loss(gt_pcl, pred_rot, 'chamfer')

optim = tf.train.AdamOptimizer(lr).minimize(1e4*loss, var_list=angles)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # Optimize for angles using point cloud based loss (Chamfer distance)
    for iters in range(N_ITERS):
        feed_dict = get_feed_dict(iters, models_list)
        _loss, _angles, _ = sess.run([loss, angles, optim], feed_dict)

        if iters % print_n == 0:
            print _loss
            print _angles * 180. / np.pi

    if save_outputs:
        for iters in range(N_MODELS):
            if iters % 100 == 0:
                print iters, '/', N_MODELS
            feed_dict = get_feed_dict(iters, models_list)
            model_name = models_list[iters].split('/')[-1].split('_')[1]
            _pred_pcl = sess.run(pred_rot[0], feed_dict)
            np.savetxt(join(out_dir, model_name + '_pred.xyz'), _pred_pcl)
            np.savetxt(join(out_dir, model_name + '_gt.xyz'),
                    feed_dict[gt_pcl][0])
            np.save(join(out_dir, model_name+'_pred.npy'), _pred_pcl)

        _rotmat, _angles = sess.run([rot_mat, angles], feed_dict)
        np.savetxt(join(exp_dir, 'angles.txt'), _angles * 180. / np.pi)
        np.save(join(exp_dir, 'angles.npy'), _angles * 180. / np.pi)
        print sess.run(rotmat_gt)
    print sess.run(rot_mat, feed_dict)
