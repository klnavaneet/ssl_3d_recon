################################################
# Various helper functions grouped together
################################################

import os, sys
sys.path.append('../chamfer_utils')
from os.path import join
import pdb
import re

import tensorflow as tf
import numpy as np
import show3d_balls
import cv2


def create_folder(folders_list):
    for folder in folders_list:
        if not os.path.exists(folder):
            os.makedirs(folder)


def load_model_from_ckpt(sess, saver, ckpt_folder):
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    st_iters = -1
    if ckpt is not None:
        print ('loading '+os.path.abspath(ckpt.model_checkpoint_path) + '  ....')
        saver.restore(sess, os.path.abspath(ckpt.model_checkpoint_path))
        st_iters = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))
        init_flag = False
    return st_iters+1


def load_model_from_ckpt_pose(sess, saver, ckpt_folder):
    vars_to_restore=[v for v in tf.global_variables() if "pose_net" in v.name]
    vars_to_restore_dict = {}
    for v in vars_to_restore:
        vars_to_restore_dict[v.name[:-2]] = v
    saver=tf.train.Saver(vars_to_restore_dict)
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    st_iters = -1
    if ckpt is not None:
        print ('loading pose_network '+os.path.abspath(ckpt.model_checkpoint_path) + '  ....')
        saver.restore(sess, os.path.abspath(ckpt.model_checkpoint_path))

        st_iters = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))
        init_flag = False
    return st_iters+1


def load_model_from_ckpt_recon_net(sess, saver, ckpt_folder):
    vars_to_restore=[v for v in tf.global_variables() if "recon_net" in v.name]
    vars_to_restore_dict = {}
    for v in vars_to_restore:
        vars_to_restore_dict[v.name[:-2]] = v
    saver=tf.train.Saver(vars_to_restore_dict)
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    st_iters = -1
    if ckpt is not None:
        print ('loading recon_network '+os.path.abspath(ckpt.model_checkpoint_path) + '  ....')
        saver.restore(sess, os.path.abspath(ckpt.model_checkpoint_path))

        st_iters = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))
        init_flag = False
    return st_iters+1


def average_stats(val_mean, val_batch, iters):
    '''
    Maintain running average of logged values
    '''
    val_upd = [((item*iters)+batch_item)/(iters+1) for (item, batch_item) in\
            zip(val_mean, val_batch)]
    return val_upd


def labels2img(labels_map, N_CLS, is_onehot=False, pcl=False):
    '''
    Convert part-segmentation representation from labels to a colour image
    Args:
        labels_map: float, (BS,H,W) or (BS,H,W,N_CLS+1)
                label map, either labels or one-hot encoding (includes background class)
                one-hot representation can be a probabilistic value
        N_CLS: int, number of classes
        is_onehot: boolean,
                   True if input is in one-hot representation
    Returns:
        out: float, (BS,H,W,3)
             output colour coded map
    '''
    if is_onehot:
        labels_map = np.argmax(labels_map, axis=-1) # 0 is background label in labels.npy
    cc = [[0,0,0],[0,0,64],[0,128,0],[0,128,128],[128,0,0],
          [128,0,128],[128,128,0],[128,128,128]]
    BS, H, W = (labels_map.shape)[:3]
    out = np.zeros(shape=(BS,H,W,3))
    cls_indices = [];
    for k in range(N_CLS+1):
        cls_indices.append(labels_map==k)
        out[cls_indices[k]==True] = cc[k]
    return out


def img2labels(ip, N_CLS, is_onehot=False, pcl=False):
    '''
    inputs and outputs are numpy arrays
    Convert color-coded image/pcl to part-segmentation labels
    args:
        labels_map: float, (BS,H,W) or (BS,H,W,N_CLS+1)
                label map, either labels or one-hot encoding (includes background class)
                one-hot representation can be a probabilistic value
        N_CLS: int, number of classes
        is_onehot: boolean,
                   True if input is in one-hot representation
    returns:
        out: float, (BS,H,W,3)
             output colour coded map
    '''
    cc = [[0,0,0],[0,0,64],[0,0,128],[0,128,0],[0,128,128],[128,0,0],
            [128,0,128],[128,128,0],[128,128,128]]
    ip_shape = ip.shape
    if not pcl:
        B, H, W, _ = ip_shape
        ip = np.reshape(ip, (B*H*W, 3))
    else:
        B, N_PTS, _ = ip_shape
        ip = np.reshape(ip, (B*N_PTS, 3))

    ip_lbl = np.zeros((len(ip)), dtype=np.uint8); indices = {};
    ip_list = ip.tolist()
    for cls in range(N_CLS):
        indices[cls] = [idx for idx in range(len(ip)) if ip_list[idx]==cc[cls]]
        ip_lbl[indices[cls]] = cls
    if not pcl:
        ip_lbl = np.reshape(ip_lbl, (B,H,W))
    else:
        ip_lbl = np.reshape(ip_lbl, (B,N_PTS))

    return ip_lbl

def image_gradients(image):
  # From official tensorflow implementation of tf.image.image_gradients
  """Returns image gradients (dy, dx) for each color channel.
  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.
  Arguments:
    image: Tensor with shape [batch_size, h, w, d].
  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).
  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
  if image.get_shape().ndims != 4:
    raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not %s.', image.get_shape())
  image_shape = tf.shape(image)
  batch_size, height, width, depth = tf.unstack(image_shape)
  dy = image[:, 1:, :, :] - image[:, :-1, :, :]
  dx = image[:, :, 1:, :] - image[:, :, :-1, :]

  # Return tensors with same size as original image by concatenating
  # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
  shape = tf.stack([batch_size, 1, width, depth])
  dy = tf.concat([dy, tf.zeros(shape, image.dtype)], 1)
  dy = tf.reshape(dy, image_shape)

  shape = tf.stack([batch_size, height, 1, depth])
  dx = tf.concat([dx, tf.zeros(shape, image.dtype)], 2)
  dx = tf.reshape(dx, image_shape)

  return dy, dx


def tf_rotate(xyz, xangle=0, yangle=0, inverse=False):
    '''
    Rotate point cloud by given values of x and y angles.
    Args:
        xyz: (BS,N_PTS,3); input pcl tensor
        xangle: angle to rotate about x-axis
        yangle: angle to rotate about y-axis
    Returns:
        xyz: (BS,N_PTS,3); Rotated point cloud
    '''
    batch_size = (xyz.shape)[0]
    rotmat = np.eye(3)

    rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))

    rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
    _rotmat = tf.constant(rotmat, dtype=tf.float32)
    _rotmat = tf.reshape(tf.tile(_rotmat,(batch_size,1)),
            shape=(batch_size,3,3))
    return tf.matmul(xyz, _rotmat)


def np_rotate(xyz, xangle=0, yangle=0, inverse=False):
    '''
    Rotate point cloud by given values of x and y angles.
    Args:
        xyz: (N_PTS,3); input pcl numpy array
        xangle: angle to rotate about x-axis
        yangle: angle to rotate about y-axis
    Returns:
        xyz: (N_PTS,3); Rotated point cloud
    '''
    rotmat = np.eye(3)

    rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))

    rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))

    if inverse:
            rotmat = np.linalg.inv(rotmat)

    return xyz.dot(rotmat)


def save_screenshots(_gt_scaled, _pr_scaled, img, screenshot_dir, fid, eval_set,
        rgb=False, rgb_feats=None, ballradius=3):
    '''
    Save point cloud results as a series of projections from different angles.
    '''

    # clock, front, anticlock, side, back, top
    #xangles = np.array([-50, 0, 50, 90, 180, 0]) * np.pi / 180.
    #yangles = np.array([20, 20, 20, 20, 20, 90]) * np.pi / 180.

    xangles = np.array([-50,-30,-10,10,30,50,70,90,110,130,150,170,190,210,230,250,270]) * np.pi / 180.
    yangles = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])* np.pi / 180.
    gts = []
    results = []
    overlaps = []

    for xangle, yangle in zip(xangles, yangles):
            if rgb:
                gt_rot = show3d_balls.get2Drgb(np_rotate(_gt_scaled, xangle=xangle, yangle=yangle), rgb_feats[0], ballradius=ballradius)
                result_rot = show3d_balls.get2Drgb(np_rotate(_pr_scaled, xangle=xangle, yangle=yangle), rgb_feats[1], ballradius=ballradius)
                overlap_rot = show3d_balls.get2Dtwopoints_rgb(np_rotate(_gt_scaled, xangle=xangle, yangle=yangle), np_rotate(_pr_scaled, xangle=xangle, yangle=yangle), rgb_feats[0], rgb_feats[1], ballradius=ballradius)
            else:
                gt_rot = show3d_balls.get2D(np_rotate(_gt_scaled, xangle=xangle, yangle=yangle), ballradius=ballradius)
                result_rot = show3d_balls.get2D(np_rotate(_pr_scaled, xangle=xangle, yangle=yangle), ballradius=ballradius)
                overlap_rot = show3d_balls.get2Dtwopoints(np_rotate(_gt_scaled, xangle=xangle, yangle=yangle), np_rotate(_pr_scaled, xangle=xangle, yangle=yangle), ballradius=ballradius)
            gts.append(gt_rot)
            results.append(result_rot)
            overlaps.append(overlap_rot)
    transparent_img=cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    transparent_img[np.all(transparent_img==[0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    cv2.imwrite(join(screenshot_dir, '%s_%s_inp.png'%(eval_set, fid)), img)
    cv2.imwrite(join(screenshot_dir, '%s_%s_inp_alpha_0.png'%(eval_set, fid)), transparent_img)
    gt = np.concatenate(gts, 1)
    result = np.concatenate(results, 1)
    final = np.concatenate((gt, result), 0)
    mask = np.sum(final, axis=-1, keepdims=True)
    mask = ((mask>0).astype(final.dtype))*final.max()
    final = np.concatenate((final, mask), axis=-1)
    if rgb:
        cv2.imwrite(join(screenshot_dir, '%s_%s.png'%(eval_set,fid)), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        transparent_im=cv2.cvtColor(final, cv2.COLOR_RGB2RGBA)
        transparent_im[np.all(transparent_im==[0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
        cv2.imwrite(join(screenshot_dir, '%s_%s_alpha_0.png'%(eval_set,fid)), transparent_im)

    else:
        cv2.imwrite(join(screenshot_dir, '%s_%s.png'%(eval_set,fid)), final)

    save_gifs = True
    if save_gifs:
            import imageio
            final = [np.concatenate((gt,result), 1) for gt,result in zip(gts,results)]
            imageio.mimsave(join(screenshot_dir, '%s_%s.gif'%(eval_set,fid)), final, 'GIF', duration=0.3)

    return


def save_outputs(out_dir, iters, feed_dict, img_name):
    _img, _mask, pose, _pose_out = sess.run([img_out, mask_out, pose_all[0], pose_out[1:]], feed_dict)
    _img = np.stack(_img, axis=1)[0]
    _mask = np.stack(_mask, axis=1)[0]
    # Normalize to [0,255]
    _img = _img*255
    _mask = _mask*255

    sc.imsave('%s/%d_%s_gt_pose_%d_%d.png'%(out_dir, iters, img_name[0],
        pose[0,0]*(180./np.pi), pose[0,1]*(180./np.pi)), feed_dict[img_ip][0]*255)
    sc.imsave('%s/%d_%s_gt_pose_%d_%d_mask.png'%(out_dir, iters, img_name[0],
        pose[0,0]*(180./np.pi), pose[0,1]*(180./np.pi)), feed_dict[mask_ip][0]*255)
    for i in range(FLAGS.N_PROJ):
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
    for i in range(FLAGS.batch_size):
        np.save('%s/%d_%s_pcl.npy'%(out_dir, iters, img_name[i]), _pcl[i])
        np.savetxt('%s/%d_%s_pcl.xyz'%(out_dir, iters, img_name[i]), _pcl[i])
    return _pcl


def remove_outliers(pcl, min_val=-.5, max_val=0.5):
    '''
    Remove outlier points in pcl and replace with existing points --> used only
    during visualization, SHOULD NOT be used during metric calculation
    Args:
            pcl: float, (BS,N_PTS,3); input point cloud with outliers
            min_val, max_val: float, (); minimum and maximum value of the
                        co-ordinates, beyond which point is treated as outlier
    Returns:
            pcl: float, (BS,N_PTS,3); cleaned point cloud
    '''
    pcl_clip = np.clip(pcl, min_val, max_val)
    indices = np.equal(pcl, pcl_clip)
    ind, _ = np.where(indices!=True)
    pcl[ind] = pcl[0]
    return pcl
