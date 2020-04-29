##############################################################################
# Main training code for OURS-CC model - Nearest Neighbour loss is not included
##############################################################################

import os, sys
from os.path import join, abspath, basename
import argparse
import time
import json
import pdb
import pprint

import tensorflow as tf
import numpy as np
import scipy.misc as sc

from src.chamfer_utils import tf_nndistance
from src.proj_codes import (rgb_cont_proj as get_proj_rgb, perspective_transform,
                            cont_proj as get_proj_mask, world2cam)
from src.net_archs import recon_net_tiny_rgb_skipconn as recon_net, pose_net
from src.get_losses import get_3d_loss, get_img_loss, get_pose_loss, get_chamfer_dist
from src.dataloader import fetch_data
from src.utils.helper_funcs import create_folder, load_model_from_ckpt, average_stats
from src.shapenet_taxonomy import shapenet_category_to_id, shapenet_id_to_category


parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, required=True,
        help='Name of the experiment')
parser.add_argument('--gpu', type=int, default=1,
        help='gpu id to be used')
parser.add_argument('--dataset', type=str, default='shapenet_train',
        help='dataset to be used: [shapenet_train, pfcn]')
parser.add_argument('--use_gt_pose', action='store_true',
        help='use GT pose from input view to train the network')
parser.add_argument('--optimise_pose', action='store_true',
        help='optimise the pose network')
parser.add_argument('--overfit', action='store_true',
        help='train with just a single data instance - overfit the model')
parser.add_argument('--load_model', action='store_true',
        help='load model weights from checkpoint, True if argument is present')
parser.add_argument('--affinity_loss', action='store_true',
        help='use affinity loss for masks to train the network')
parser.add_argument('--symmetry_loss', action='store_true',
        help='use symmetry loss for masks to train the network')
parser.add_argument('--loss', type=str, default='bce',
        help='use either bce loss or bce with logits loss(i.e treat projections\
        as logits), [bce_prob, bce]')
parser.add_argument('--_3d_loss_type', type=str, default='init_model',
        help='way to choose pairs for 3d consistency loss. adj_model-choose\
        next index, init_model-always choose original input reconstruction')
parser.add_argument('--N_ITERS', type=int, default=100001,
        help='Number of iterations to run the experiment')
parser.add_argument('--batch_size', type=int, default=1,
        help='batch size to be used')
parser.add_argument('--H', type=int, default=64,
        help='height of input images')
parser.add_argument('--W', type=int, default=64,
        help='width of input images')
parser.add_argument('--bottleneck', type=int, default=128,
        help='dimension of bottleneck layer')
parser.add_argument('--N_PTS', type=int, default=1024,
        help='dimension of output point cloud')
parser.add_argument('--N_PROJ', type=int, default=2,
        help='number of projections for each input image')
parser.add_argument('--categ', type=str, default='car',
        help='category to be used for training')
parser.add_argument('--sigma_sq', type=float, default=0.4,
        help='variance of mask projection gaussian function')
parser.add_argument('--lr', type=float, default=1e-6,
        help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
        help='beta1 parameter in Adam Optimizer')
parser.add_argument('--print_n', type=int, default=100,
        help='print losses every print_n iterations')
parser.add_argument('--save_n', type=int, default=1000,
        help='save sample outputs every save_n iterations')
parser.add_argument('--save_model_n', type=int, default=5000,
        help='save model weights every save_model_n iterations')

parser.add_argument('--lambda_ae', type=float, default=1.,
        help='Weight for image auto-encoding loss')
parser.add_argument('--lambda_ae_pose', type=float, default=1.,
        help='Weight for image auto-encoding loss for pose net')
parser.add_argument('--lambda_ae_mask', type=float, default=1.,
        help='Weight for mask auto-encoding loss')
parser.add_argument('--lambda_mask_fwd', type=float, default=1.,
        help='Weight for mask 2d chamfer(affinity) fwd loss')
parser.add_argument('--lambda_mask_bwd', type=float, default=1.,
        help='Weight for mask 2d chamfer(affinity) bwd loss')
parser.add_argument('--lambda_mask_pose', type=float, default=1.,
        help='Weight for mask auto-encoding loss for pose net')
parser.add_argument('--lambda_3d', type=float, default=1.,
        help='Weight for 3D consistency loss')
parser.add_argument('--lambda_pose', type=float, default=0.,
        help='Weight for pose loss')
parser.add_argument('--lambda_symm', type=float, default=0.,
        help='Weight for symmetry loss')

args = parser.parse_args()

print '*'*50
pprint.pprint(args)
print '*'*50

if args.use_gt_pose:
    print 'GT Pose'

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

categ = shapenet_category_to_id[args.categ]
mode = 'train'

print 'Full Shapnet Data'
data_dir = './data/ShapeNet_rendered/%s' % categ
tfrecords_file_rgb = './data/%s_%s_image.tfrecords' % (categ, mode)
tfrecords_file_mask = './data/%s_%s_mask.tfrecords' % (categ, mode)
tfrecords_file_pose = './data/%s_%s_pose.tfrecords' % (categ, mode)

if not args.use_gt_pose:
    tfrecords_file = {'rgb': tfrecords_file_rgb, 'mask': tfrecords_file_mask}
    dtypes = ['rgb', 'mask']
else:
    tfrecords_file = {'rgb': tfrecords_file_rgb, 'mask': tfrecords_file_mask,
            'pose': tfrecords_file_pose}
    dtypes = ['rgb', 'mask', 'pose']


models = np.load('splits/images_list_%s_%s.npy'%(categ, mode))
shuffle_len = len(models)
print 'Train Categ: ', shapenet_id_to_category[categ], 'Train Models: ', shuffle_len

# Log directories
BASE_DIR = './'
exp_dir = join(BASE_DIR, args.exp)
ckpt_dir = join(BASE_DIR, args.exp, 'checkpoints')
logs_dir = join(BASE_DIR, args.exp, 'logs')
log_file = join(args.exp, 'logs.txt')
proj_images_dir = join(BASE_DIR, args.exp, 'log_proj_images')
proj_pcl_dir = join(BASE_DIR, args.exp, 'log_proj_pcl')
create_folder([ckpt_dir, logs_dir, proj_images_dir,
               proj_pcl_dir])

filename = basename(__file__)
os.system('cp %s %s'%(filename, exp_dir))
if args.categ == 'car':
    filename_bash = 'run.sh'
else:
    filename_bash = 'chair_run.sh'
os.system('cp %s %s'%(filename_bash, exp_dir))
args_file = join(logs_dir, 'args.json')
with open(args_file, 'w') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)


def save_outputs(out_dir, iters, feed_dict, img_name):
    _img, _mask, pose, _pose_out = sess.run([img_out, mask_out, pose_all[0],
                                             pose_out[1:]], feed_dict)
    _img = np.stack(_img, axis=1)[0]
    _mask = np.stack(_mask, axis=1)[0]
    # Normalize to [0,255]
    _img = _img*255
    _mask = _mask*255

    sc.imsave('%s/%d_%s_gt_pose_%d_%d.png'%(out_dir, iters, img_name[0],
        pose[0,0]*(180./np.pi), pose[0,1]*(180./np.pi)), feed_dict[img_ip][0]*255)
    sc.imsave('%s/%d_%s_gt_pose_%d_%d_mask.png'%(out_dir, iters, img_name[0],
        pose[0,0]*(180./np.pi), pose[0,1]*(180./np.pi)), feed_dict[mask_ip][0]*255)
    for i in range(args.N_PROJ):
        sc.imsave('%s/%d_%s_pred_%d_pose_%d_%d.png'%(out_dir, iters, img_name[0],
            i, pose[i,0]*(180./np.pi), pose[i,1]*(180./np.pi)), _img[i])
        sc.imsave('%s/%d_%s_pred_%d_pose_%d_%d_mask.png'%(out_dir, iters, img_name[0],
            i, pose[i,0]*(180./np.pi), pose[i,1]*(180./np.pi)), _mask[i])
        print 'Px: %03d, Prx: %03d, Py: %03d, Pry: %03d' % (pose[i,0]*(180./np.pi),
            _pose_out[i][0,0]*(180./np.pi), pose[i,1]*(180./np.pi),
            _pose_out[i][0,1]*(180./np.pi))
    return True


def save_outputs_pcl(out_dir, iters, feed_dict, img_name):
    _pcl = sess.run([pcl_out[0], pcl_rgb_out[0]], feed_dict)
    _pcl = np.concatenate(_pcl)
    for i in range(args.batch_size):
        np.save('%s/%d_%s_pcl.npy'%(out_dir, iters, img_name[i]), _pcl[i])
        np.savetxt('%s/%d_%s_pcl.xyz'%(out_dir, iters, img_name[i]), _pcl[i])
    return _pcl

# Create Placeholders
img_ip = tf.placeholder(tf.float32, (args.batch_size, args.H, args.W, 3),
        name='input_image')
mask_ip = tf.placeholder(tf.float32, (args.batch_size, args.H, args.W),
        name='input_mask')
pose_ip = tf.placeholder(tf.float32, (args.batch_size, args.N_PROJ-1, 2),
        name='input_pose')
if args.use_gt_pose:
    pose_gt = tf.placeholder(tf.float32, (args.batch_size, 2),
            name='gt_pose')

train_loss_summ = []
loss_names = ['Loss_total', 'Loss_ae', 'Loss_mask', '2D_Ch_Fwd', '2D_Ch_Bwd',
        'Loss_3D', 'Loss_pose', 'Loss_symm']
for idx, name in enumerate(loss_names):
    train_loss_summ.append(tf.placeholder(tf.float32, (),
        name=name))

pcl_out = []; pose_out = []; img_out = []; pcl_rgb_out = [];
pcl_out_rot = []; pcl_out_persp = []; mask_out = [];

with tf.variable_scope('recon_net'):
    pcl_xyz, pcl_rgb = recon_net(img_ip, args)
pcl_out.append(pcl_xyz)
pcl_rgb_out.append(pcl_rgb)

if not args.use_gt_pose:
    with tf.variable_scope('pose_net'):
        pose_out.append(pose_net(img_ip, args))
else:
    pose_out.append(pose_gt)
    # Dummy - for code compatibility
    with tf.variable_scope('pose_net'):
        temp = pose_net(img_ip, args)
pose_all = tf.concat([tf.expand_dims(pose_out[0], axis=1), pose_ip], axis=1)

# Perspective projection - 1.Change from world to camera co-ordinates in the
# given view-point 2.Do # perspective transformation of the point cloud
# 3. Project the transformed pcl onto the 2D plane
for idx in range(args.N_PROJ):
    pcl_out_rot.append(world2cam(pcl_out[0], pose_all[:, idx, 0],
        pose_all[:,idx,1], 2., 2., args.batch_size))
    pcl_out_persp.append(perspective_transform(pcl_out_rot[idx],
        args.batch_size))
    img_out.append(get_proj_rgb(pcl_out_persp[idx], pcl_rgb_out[0], args.N_PTS,
        args.H, args.W)[0])
    mask_out.append(get_proj_mask(pcl_out_persp[idx], args.H, args.W,
        args.N_PTS, args.sigma_sq))

# Reconstruct the point cloud from and predict the pose of projected images
for idx in range(args.N_PROJ):
    with tf.variable_scope('recon_net', reuse=True):
        pcl_xyz, pcl_rgb = recon_net(img_out[idx], args)
    pcl_out.append(pcl_xyz)
    pcl_rgb_out.append(pcl_rgb)

    with tf.variable_scope('pose_net', reuse=True):
        pose_out.append(pose_net(img_out[idx], args))

# Define Losses
# 2D Consistency Loss - L2
img_ae_loss, _, _ = get_img_loss(img_ip, img_out[0], 'l2_sq')
mask_ae_loss, mask_fwd, mask_bwd = get_img_loss(mask_ip, mask_out[0],
        args.loss, affinity_loss=args.affinity_loss)

# 3D Consitency Loss
consist_3d_loss = 0.
for idx in range(args.N_PROJ):
    if args._3d_loss_type == 'adj_model':
        consist_3d_loss += get_3d_loss(pcl_out[idx], pcl_out[idx+1], 'chamfer')
    elif args._3d_loss_type == 'init_model':
        consist_3d_loss += get_3d_loss(pcl_out[idx], pcl_out[0], 'chamfer')

# Pose Loss
pose_loss_pose = get_pose_loss(pose_ip, tf.stack(pose_out[2:], axis=1), 'l1')

# Symmetry loss - assumes symmetry of point cloud about z-axis
# Helps obtaining output aligned along z-axis
pcl_y_pos = tf.to_float(pcl_out[0][:,:,1:2]>0)
pcl_y_neg = tf.to_float(pcl_out[0][:,:,1:2]<0)
pcl_pos = pcl_y_pos*tf.concat([pcl_out[0][:,:,:1], tf.abs(pcl_out[0][:,:,1:2]),
        pcl_out[0][:,:,2:3]], -1)
pcl_neg = pcl_y_neg*tf.concat([pcl_out[0][:,:,:1], tf.abs(pcl_out[0][:,:,1:2]),
        pcl_out[0][:,:,2:3]], -1)
symm_loss = get_chamfer_dist(pcl_pos, pcl_neg)[-1]

# Total Loss
loss = (args.lambda_ae*img_ae_loss) + (args.lambda_3d*consist_3d_loss) +\
        (args.lambda_pose*pose_loss_pose)
recon_loss = (args.lambda_ae*img_ae_loss) + (args.lambda_3d*consist_3d_loss)\
                + (args.lambda_ae_mask*mask_ae_loss) +\
                (args.lambda_mask_fwd*mask_fwd) + (args.lambda_mask_bwd*mask_bwd)
if args.symmetry_loss:
    recon_loss += (args.lambda_symm*symm_loss)
pose_loss = (args.lambda_ae_pose*img_ae_loss) + (args.lambda_pose*pose_loss_pose)\
                + (args.lambda_mask_pose*mask_ae_loss)

# Optimizer
recon_vars = [var for var in tf.global_variables() if 'recon' in var.name]
pose_vars = [var for var in tf.global_variables() if 'pose' in var.name]
optim_recon = tf.train.AdamOptimizer(args.lr, args.beta1).minimize(recon_loss,
        var_list=recon_vars)
if not args.use_gt_pose:
    optim_pose = tf.train.AdamOptimizer(args.lr, args.beta1).minimize(pose_loss,
            var_list=pose_vars)

# Add tensorboard summaries
loss_summ = []
for idx, name in enumerate(loss_names):
    loss_summ.append(tf.summary.scalar(name, train_loss_summ[idx]))
train_summ = tf.summary.merge(loss_summ)

# Define savers to load and store models
saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    train_writer = tf.summary.FileWriter(logs_dir, sess.graph_def)
    sess.run(tf.global_variables_initializer())
    if args.load_model:
        st_iters = load_model_from_ckpt(sess, saver, ckpt_dir)
        if st_iters !=0:
            print 'model_loaded'
    else:
        st_iters = 0

    # Load input data
    next_element = fetch_data(tfrecords_file, args.batch_size,
            shuffle_len, dtype=dtypes)

    def get_feed_dict():
        if args.use_gt_pose:
            img_prop, mask_prop, pose_prop = sess.run(next_element)
            pose, _ = pose_prop
        else:
            img_prop, mask_prop = sess.run(next_element)
        img, img_name = img_prop
        mask, _ = mask_prop
        # Normalize to [0,1]
        img = img.astype(np.float32)/255.
        mask = mask.astype(np.float32)/255.

        # Sample angles: azimuth-->[-180,180], elevation-->[-20,40]
        # Convert angles from deg to rad
        _pose_ip = np.random.rand(args.batch_size, args.N_PROJ-1, 2)
        _pose_ip[:,:,0] = (_pose_ip[:,:,0]*2*np.pi)-(np.pi)
        _pose_ip[:,:,1] = (_pose_ip[:,:,1]*(60.*np.pi/180.))-(20./180.*np.pi)
        if not args.use_gt_pose:
            feed_dict = {img_ip: img, pose_ip: _pose_ip, mask_ip: mask[:,:,:,0]}
        else:
            feed_dict = {img_ip: img, pose_ip: _pose_ip, mask_ip: mask[:,:,:,0],
                    pose_gt: pose}
        return feed_dict, img_name

    # Constant input to overfit model
    feed_dict_of, img_name_of = get_feed_dict()

    if st_iters == 0:
        print_str = 'Iters   Total     2D      Mask    Mask_fwd    Mask_bwd     3D     Pose    Symm  Time \n'
        with open(log_file, 'w') as f:
            f.write(print_str)

    time_st = time.time()
    batch_out_mean = [0.] * len(loss_names)
    for iters in range(st_iters, args.N_ITERS):
        if not args.overfit:
            feed_dict, img_name = get_feed_dict()
        else:
            feed_dict, img_name = feed_dict_of, img_name_of

        # Network training
        if not args.use_gt_pose and args.optimise_pose:
            batch_out = sess.run([loss, img_ae_loss, mask_ae_loss, mask_fwd,
                mask_bwd, consist_3d_loss, pose_loss_pose, symm_loss,
                optim_recon, optim_pose], feed_dict)
            # Use averaged loss values for logging
            batch_out_mean = average_stats(batch_out_mean, batch_out[:-2],
                iters%args.print_n)
        else:
            batch_out = sess.run([loss, img_ae_loss, mask_ae_loss, mask_fwd,
                mask_bwd, consist_3d_loss, pose_loss_pose, symm_loss,
                optim_recon], feed_dict)
            # Use averaged loss values for logging
            batch_out_mean = average_stats(batch_out_mean, batch_out[:-1],
                iters%args.print_n)
        _loss, _ae_loss, _mask_ae_loss, _mask_fwd, _mask_bwd, _3d_loss, _pose_loss, _symm_loss = batch_out_mean

        if (iters + 1) % args.print_n == 0:
            feed_dict_summ = {}
            for i, item in enumerate(batch_out_mean):
                feed_dict_summ[train_loss_summ[i]] = item
            summ = sess.run(train_summ, feed_dict_summ)
            print 'Iters:%d, Total: %.4f, 2D: %.4f, Mask: %.4f, 2D_Ch_F: %.4f, 2D_Ch_B: %.4f, 3D: %.4f, Pose: %.4f, Symm: %.4f T: %d'\
                % (iters, _loss, _ae_loss, _mask_ae_loss, _mask_fwd, _mask_bwd,
                  _3d_loss, _pose_loss, _symm_loss, (time.time()-time_st)//60.)

            # Log loss values in file
            print_str = '%06d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %04d \n'\
                % (iters, _loss, _ae_loss, _mask_ae_loss, _mask_fwd, _mask_bwd,
                  _3d_loss, _pose_loss, _symm_loss, (time.time()-time_st)//60.)
            with open(log_file, 'a') as f:
                f.write(print_str)

            # Add to tensorboard summary
            train_writer.add_summary(summ, iters)

        if iters % args.save_n == 0:
            # Save image outputs
            save_outputs(proj_images_dir, iters, feed_dict, img_name)
            save_outputs_pcl(proj_pcl_dir, iters, feed_dict, img_name)

        if iters % args.save_model_n == 0 and iters != 0:
            # Save Model checkpoint
            saver.save(sess, join(ckpt_dir, 'model'), global_step=iters)
            print 'Model Saved at iter: ', iters
