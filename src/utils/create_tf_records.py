#########################################################
# Create tensorflow records file for (projected) images
#########################################################

import os, sys
from os.path import join, abspath, basename
import glob
import time
import pdb

import tensorflow as tf
import numpy as np

# Change these values based on the data for which tfrecords file needs to be
# created (do them sequentially - set only one of them to True at a time).
save_img = True
save_pcl = False
save_pose = False

dataset = 'shapenet'
if save_img or save_pose:
    data_dir = '../../data/ShapeNet_rendered'
elif save_pcl:
    data_dir = '../../data/ShapeNet_v1'
out_dir = '../../data'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

categs = ['02691156', '02958343', '03001627']
modes = ['train', 'val', 'test']

def read_img(img_path, cnt=None):
    # Read only one of the projections for each model. Randomly choose the one
    # out of 10 existing projections
    if cnt==None:
        cnt = np.random.randint(0,10)
    if save_img:
        with open(join(img_path, 'render_%s.png'%cnt)) as f:
            img_bytes = f.read()
    elif save_mask:
        with open(join(img_path, 'mask_%s.png'%cnt)) as f:
            img_bytes = f.read()

    img_name = img_path.split('/')[-1]+'_%s'%cnt
    example = tf.train.Example(features = tf.train.Features(feature = { \
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


def read_pose(img_path, cnt=None):
    # Read only one of the projections for each model. Randomly choose one
    # out of 10 existing projections or use the provided id
    if cnt==None:
        cnt = np.random.randint(0,10)

    with open(join(img_path, 'view.txt'), 'r') as fp:
        angles = [item.split('\n')[0] for item in fp.readlines()]
    angle = angles[int(cnt)]
    angle = [float(item)*(np.pi/180.) for item in angle.split(' ')[:2]]
    pose_bytes = np.array(angle).tostring()

    img_name = img_path.split('/')[-1]+'_%s'%cnt
    example = tf.train.Example(features = tf.train.Features(feature = { \
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
        'pose': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


def read_pcl(pcl_path):
    pcl = np.load(join(pcl_path, 'pcl_1024_fps_trimesh_colors.npy'))
    pcl_bytes = pcl.tostring()
    pcl_name = pcl_path.split('/')[-1]
    example = tf.train.Example(features = tf.train.Features(feature = { \
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_name])),
        'pcl': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


for mode in modes:
    for categ in categs:
        print categ
        overwrite = True

        models = np.load('../../splits/images_list_%s_%s.npy'%(categ, mode))
        image_ids = [model[0].split('_')[1] for model in models]
        models = [model[0].split('_')[0] for model in models]
        if save_img:
            if dataset == 'pfcn':
                out_file = '%s_%s_image_pfcn.tfrecords'%(join(out_dir, categ),
                        mode)
            else:
                out_file = '%s_%s_image.tfrecords'%(join(out_dir, categ), mode)
        elif save_pcl:
            out_file = '%s_%s_pcl_color.tfrecords'%(join(out_dir, categ), mode)
        elif save_pose:
            out_file = '%s_%s_pose.tfrecords'%(join(out_dir, categ), mode)
        elif save_mask:
            out_file = '%s_%s_mask.tfrecords'%(join(out_dir, categ), mode)

        if os.path.exists(out_file):
            response = raw_input('File exists! Replace? (y/n): ')
            if response != 'y':
                overwrite = False

        N_missing = 0
        time_st = time.time()
        if overwrite:
            with tf.python_io.TFRecordWriter(out_file) as writer:
                for idx, model in enumerate(models):
                    if idx%500==0:
                        print idx, '/', len(models)
                        print 'Time: ', (time.time() - time_st)//60
                    try:
                        if save_img:
                            img_str = read_img(join(data_dir, categ, model),
                                    image_ids[idx])
                            writer.write(img_str)
                        elif save_pose:
                            pose_str = read_pose(join(data_dir, categ, model),
                                    image_ids[idx])
                            writer.write(pose_str)
                        elif save_pcl:
                            pcl_str = read_pcl(join(data_dir, categ, model))
                            writer.write(pcl_str)
                    except KeyboardInterrupt:
                        sys.exit()
                    except:
                        N_missing += 1
                        continue
            print 'Time: ', (time.time() - time_st)//60
            print 'N_missing: ', N_missing
