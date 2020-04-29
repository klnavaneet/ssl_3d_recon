##############################################################################
#     Load pre-trained model and save output point clouds for evaluation
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

sys.path.append('../../src')
sys.path.append('../../src/utils')
sys.path.append('../../src/chamfer_utils')
from chamfer_utils import tf_nndistance
from proj_codes import rgb_cont_proj as get_proj_rgb, cont_proj as get_proj_mask, world2cam, perspective_transform
from net_archs import recon_net_tiny_rgb_skipconn as recon_net, pose_net
from get_losses import get_3d_loss, get_img_loss, get_pose_loss
from dataloader import fetch_data
from helper_funcs import create_folder, load_model_from_ckpt, average_stats
from shapenet_taxonomy import shapenet_category_to_id, shapenet_id_to_category


parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, required=True,
        help='Name of the experiment')
parser.add_argument('--gpu', type=int, default=1,
        help='gpu id to be used')
parser.add_argument('--dataset', type=str, default='shapenet',
        help='dataset to be used: [shapenet, pfcn]')
parser.add_argument('--mode', type=str, default='test',
        help='mode to be used: [train, val, test]')
parser.add_argument('--pred_disp', action='store_true',
        help='displacement prediction based network architecture')
parser.add_argument('--use_gt_pose', action='store_true',
        help='use GT pose from input view to train the network')
parser.add_argument('--overfit', action='store_true',
        help='train with just a single data instance - overfit the model')
parser.add_argument('--load_model', action='store_true',
        help='load model weights from checkpoint, True if argument is present')
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
parser.add_argument('--lambda_mask_pose', type=float, default=1.,
        help='Weight for mask auto-encoding loss for pose net')
parser.add_argument('--lambda_3d', type=float, default=1.,
        help='Weight for 3D consistency loss')
parser.add_argument('--lambda_pose', type=float, default=0.,
        help='Weight for pose loss')

args = parser.parse_args()

print '*'*50
pprint.pprint(args)
print '*'*50
if args.use_gt_pose:
    print 'GT Pose'

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

categ = shapenet_category_to_id[args.categ]
mode = args.mode

print 'Full Shapnet Data'
data_dir = '../../data/ShapeNet_rendered/%s'%categ
tfrecords_file_rgb = '../../data/%s_%s_image.tfrecords'%(categ, mode)
tfrecords_file_mask = '../../data/%s_%s_mask.tfrecords'%(categ, mode)
tfrecords_file_pose = '../../data/%s_%s_pose.tfrecords'%(categ, mode)

if not args.use_gt_pose:
    tfrecords_file = {'rgb': tfrecords_file_rgb, 'mask': tfrecords_file_mask}
    dtypes = ['rgb', 'mask']
else:
    tfrecords_file = {'rgb': tfrecords_file_rgb, 'mask': tfrecords_file_mask,
            'pose': tfrecords_file_pose}
    dtypes = ['rgb', 'mask', 'pose']

models = np.load('../../splits/images_list_%s_%s.npy'%(categ, mode))
shuffle_len = len(models)
print 'Categ: ', shapenet_id_to_category[categ], 'Models: ', shuffle_len

# Log directories
BASE_DIR = './'
exp_dir = join(BASE_DIR, args.exp)
ckpt_dir = join(BASE_DIR, args.exp, 'checkpoints')
logs_dir = join(BASE_DIR, args.exp, 'logs')
log_file = join(args.exp, 'logs_%s.txt'%mode)
if not args.pred_disp:
    proj_images_dir = join(BASE_DIR, args.exp, 'log_proj_images_%s'%mode)
    proj_pcl_dir = join(BASE_DIR, args.exp, 'log_proj_pcl_%s'%mode)
else:
    proj_images_dir = join(BASE_DIR, args.exp, 'log_proj_images_disp_%s'%mode)
    proj_pcl_dir = join(BASE_DIR, args.exp, 'log_proj_pcl_disp_%s'%mode)
proj_pose_dir = join(BASE_DIR, args.exp, 'log_proj_pose_%s'%mode)
create_folder([ckpt_dir, logs_dir, proj_images_dir,
    proj_pcl_dir, proj_pose_dir])

filename = basename(__file__)
os.system('cp %s %s'%(filename, exp_dir))
filename_bash = 'save_pcl.sh'
os.system('cp %s %s'%(filename_bash, exp_dir))
args_file = join(logs_dir, 'args_%s.json'%mode)
with open(args_file, 'w') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)


def save_outputs(out_dir, iters, feed_dict, img_name):
    _img, _mask, pose, _pose_out = sess.run([img_out, mask_out, pose_all[0], pose_out], feed_dict)
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
    return True


def save_outputs_pose(out_dir, iters, feed_dict, img_name):
    pose = sess.run(pose_all[0], feed_dict)[0]
    np.save('%s/%d_%s_pred_pose.npy'%(out_dir, iters, img_name[0]), pose)
    np.savetxt('%s/%d_%s_pred_pose.txt'%(out_dir, iters, img_name[0]), pose)


def save_outputs_pcl(out_dir, iters, feed_dict, img_name):
    _pcl = sess.run([pcl_out[0], pcl_rgb_out[0]], feed_dict)
    _pcl = np.concatenate(_pcl, axis=2)
    for i in range(args.batch_size):
        np.save('%s/%d_%s_pcl.npy'%(out_dir, iters, img_name[i]), _pcl[i])
        np.savetxt('%s/%d_%s_pcl.txt'%(out_dir, iters, img_name[i]), _pcl[i])
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
loss_names = ['Loss_total', 'Loss_ae', 'Loss_mask', 'Loss_pose']
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

pose_all = tf.concat([tf.expand_dims(pose_out[0], axis=1)], axis=1)


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

# Define savers to load and store models
recon_vars = [var for var in tf.global_variables() if 'recon' in var.name]
pose_vars = [var for var in tf.global_variables() if 'pose' in var.name]

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
        sys.exit('model not loaded!')
        st_iters = 0

    # Load input data
    next_element = fetch_data(tfrecords_file, args.batch_size,
            shuffle_len, dtype=dtypes, shuffle=False)

    def get_feed_dict():
        if args.use_gt_pose:
            img_prop, mask_prop, pose_prop = sess.run(next_element)
            pose, _ = pose_prop
        else:
            img_prop, mask_prop = sess.run(next_element)

        img, img_name = img_prop
        mask, _ = mask_prop
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.
        mask = mask.astype(np.float32) / 255.

        if args.use_gt_pose:
            feed_dict = {img_ip: img, mask_ip: mask[:,:,:,0], pose_gt: pose}
        else:
            feed_dict = {img_ip: img, mask_ip: mask[:,:,:,0]}
        return feed_dict, img_name

    if st_iters == 0:
        print_str = 'Iters   Total     2D      Mask     Pose   Time \n'
        with open(log_file, 'w') as f:
            f.write(print_str)

    time_st = time.time()
    batch_out_mean = [0.] * 5
    for iters in range(0, shuffle_len // args.batch_size):
        if iters % 50 == 0:
            print iters, '/', shuffle_len // args.batch_size

        feed_dict, img_name = get_feed_dict()

        if iters % args.save_n == 0:
            # Save output point clouds, mask and image projections and
            # predicted pose
            save_outputs(proj_images_dir, st_iters, feed_dict, img_name)
            save_outputs_pcl(proj_pcl_dir, st_iters, feed_dict, img_name)
            save_outputs_pose(proj_pose_dir, st_iters, feed_dict, img_name)
