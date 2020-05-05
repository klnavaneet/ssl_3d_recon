###############################################################################
# Create tensorflow records file for (projected) images
# Multiple images are used as input --> all inputs are "similar to each other"
# 'Similar' images can be hardcoded or obtained as nearest neighbours based on
# the embedding distance of network trained w/o NN loss.
# k_range: determines the range of views(neighbors) from which to randomly
#          sample the NN
#        : to use first k NN, set k_range=views
###############################################################################

import os
import sys
from os.path import join, abspath, basename
import glob
import time

import tensorflow as tf
import scipy.misc as sc
import numpy as np
import random

# Set only one of them to be true at a time.
save_img = False
save_mask = True
save_pcl = False
save_pose = False

save_similar = True
load_filenames = True
dataset = 'shapenet'
# Number of nearest neighbours to save for each image
views = 5
# k_range: the number of top NN from which 'views' NN are chosen. If no random
# sampling, k_range = 1 + views, else k_range > (1 + views)
k_range = 6

if save_img or save_mask or save_pose:
    data_dir = '../../data/ShapeNet_rendered'
elif save_pcl:
    data_dir = '../../data/ShapeNet_v1'
out_dir = '../../expts/temp/enc_log_feat_trainfrom4L/latent_layer'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

sim_data_dir = '../../data/train_images/03001627/train_similar'
rand_file = ('%s/rand_file_list_%dviews.npy' % (out_dir, views))
categs = ['02691156', '02958343', '03001627']
#categs = ['02691156']

modes = ['train', 'val', 'test']
#modes = ['train']


def create_class_dict(data_dir):
    '''
    Create dictionary of image names and corresponding sub-class
    args:
            data_dir: str; path to root directory containing images of all
                    sub-classes
    returns:
            cls_dict: dict; dictionary with image names as key and
                    corresponding class id as value
            img_dict: dict; dictionary with class ids as key and list of
                    all images in the class as value
    '''

    all_cls = sorted(os.listdir(data_dir))  # all folder names
    cls_dict = {}
    img_dict = {}
    for cls_id, cls in enumerate(all_cls):
        img_names = sorted(os.listdir(join(data_dir, cls))
                           )  # names of images in folder
        img_dict[cls_id] = img_names
        for img_name in img_names:
            cls_dict[img_name] = cls_id  # like label
    return cls_dict, img_dict


def get_similar_image(image_id, image_class, class_images):
    '''
    Find image which is 'similar' to input image; similarity is defined as all
    images belonging to the same sub-class
    args:
            image_id: str; name of input image
            image_class: dict; dictionary with image names as key and
                    corresponding class id as value
            class_images: dict; dictionary with class ids as key and list of
                    all images in the class as value
    returns:
            sim_img: str; name of image similar to input
    '''
    ip_cls = image_class[image_id]
    n_img_cls = len(class_images[ip_cls])
    img_idx = np.random.randint(0, n_img_cls, 1)[0]
    sim_img = class_images[ip_cls][img_idx]
    return sim_img


def read_img(img_path, cnt=None, img2_path=None, cnt2=None):
    # Read only one of the projections for each model. Randomly choose the one
    # out of 10 existing projections
    if cnt is None:
        cnt = np.random.randint(0, 10)
    if save_img:
        with open(join(img_path, 'render_%s.png' % cnt)) as f:
            img_bytes = f.read()
        if img2_path is not None:
            img2_bytes = ["" for x in range(views - 1)]
            img2_name = ["" for x in range(views - 1)]
            for view in range(views - 1):
                with open(join(img2_path[view], 'render_%s.png' % cnt2[view])) as f:
                    img2_bytes[view] = f.read()
                img2_name[view] = img2_path[view].split(
                    '/')[-1] + '_%s' % cnt2[view]
    elif save_mask:
        with open(join(img_path, 'mask_%s.png' % cnt)) as f:
            img_bytes = f.read()
        if img2_path is not None:
            img2_bytes = ["" for x in range(views - 1)]
            img2_name = ["" for x in range(views - 1)]
            for view in range(views - 1):
                with open(join(img2_path[view], 'mask_%s.png' % cnt2[view])) as f:
                    img2_bytes[view] = f.read()
                img2_name[view] = img2_path[view].split(
                    '/')[-1] + '_%s' % cnt2[view]

    img_name = img_path.split('/')[-1] + '_%s' % cnt
    if img2_path is None:
        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
    else:
        feature = {
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}
        for view in range(views - 1):
            feat = {'filename_%d' % (view + 2): tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2_name[view]])),
                    'image_%d' % (view + 2): tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2_bytes[view]]))}
            feature.update(feat)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    example_str = example.SerializeToString()
    return example_str


def read_pose(img_path, cnt=None):
    # Read only one of the projections for each model. Randomly choose one
    # out of 10 existing projections or use the provided id
    if cnt is None:
        cnt = np.random.randint(0, 10)

    with open(join(img_path, 'view.txt'), 'r') as fp:
        angles = [item.split('\n')[0] for item in fp.readlines()]
    angle = angles[int(cnt)]
    angle = [float(item) * (np.pi / 180.) for item in angle.split(' ')[:2]]
    pose_bytes = np.array(angle).tostring()

    img_name = img_path.split('/')[-1] + '_%s' % cnt
    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
        'pose': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


def read_pcl(pcl_path):
    pcl = np.load(join(pcl_path, 'pcl_1024_fps_trimesh_colors.npy'))
    pcl_bytes = pcl.tostring()
    pcl_name = pcl_path.split('/')[-1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_name])),
        'pcl': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


if save_img:
    rand_file_list = []
else:
    rand_file_list = np.load(rand_file)
    rand_file_list_mask = []

for mode in modes:
    for categ in categs:
        print categ
        overwrite = True
        names_file = '../../data/knn_list_%s.npy' % categ
        models = np.load('../../splits/images_list_%s_%s.npy' % (categ, mode))
        image_ids = [model[0].split('_')[1] for model in models]
        models = [model[0].split('_')[0] for model in models]
        if save_similar:
            cls_dict, img_dict = create_class_dict(sim_data_dir)
        if save_img:
            if dataset == 'pfcn':
                out_file = '%s_%s_image_pfcn_knn_%dviews.tfrecords' % (join(out_dir, categ),
                                                                       mode, views)
            else:
                out_file = '%s_%s_image_similar_2img_knn_%dviews.tfrecords' % (join(out_dir,
                                                                                    categ), mode, views)
        elif save_pcl:
            out_file = '%s_%s_pcl_color_knn_%dviews.tfrecords' % (
                join(out_dir, categ), mode, views)
        elif save_pose:
            out_file = '%s_%s_pose_knn_%dviews.tfrecords' % (
                join(out_dir, categ), mode, views)
        elif save_mask:
            out_file = '%s_%s_mask_similar_2img_knn_%dviews.tfrecords' % (join(out_dir,
                                                                               categ), mode, views)

        if os.path.exists(out_file):
            response = raw_input('File exists! Replace? (y/n): ')
            if response != 'y':
                overwrite = False

        N_missing = 0
        time_st = time.time()
        if overwrite:
            with tf.python_io.TFRecordWriter(out_file) as writer:
                if save_similar:
                    # models = models*10 # models name of all models [4740]*10
                    # image_ids = image_ids*10#imageid id of image from each model[4740] *10
                    if load_filenames:
                        # 47440*2 names ['7fe64a3a70f8f6b28cd4e3ad2fcaf039_8',
                        # '26ece83dc8763b34d2b12aa6a0f050b3_9'],
                        sim_images = np.load(names_file)
                        sim_images = np.tile(
                            sim_images, (k_range, 1))  # 6 times repeat

                    sim_images_org = sim_images[:, 0]
                    models = [model.split('_')[0] for model in sim_images_org]
                    image_ids = [model.split('_')[1]
                                 for model in sim_images_org]
                    models = models  # models name of all models [4740]*10
                    # imageid id of image from each model[4740] *10
                    image_ids = image_ids
                    all_models = []

                for idx, model in enumerate(models):
                    if idx % 500 == 0:
                        print idx, '/', len(models)
                        print 'Time: ', (time.time() - time_st) // 60
                    try:
                        if save_img or save_mask:
                            if not load_filenames:
                                sim_img = get_similar_image(model + '_' +
                                                            image_ids[idx] + '.png', cls_dict, img_dict)
                                sim_model, sim_id = sim_img.split('.')[
                                    0].split('_')
                                all_models.append([model + '_' + image_ids[idx],
                                                   sim_img.split('.')[0]])
                            else:
                                if save_img:
                                    # just commented for bg images
                                    k_dist = random.sample(
                                        range(1, k_range), (views - 1))
                                    # just commented for bg images
                                    rand_file_list.append(k_dist)

                                else:
                                    k_dist = rand_file_list[idx]
                                    rand_file_list_mask.append(k_dist)

                                sim_img = ["" for x in range(views - 1)]
                                sim_model = ["" for x in range(views - 1)]
                                sim_id = ["" for x in range(views - 1)]

                                for view in range(views - 1):
                                    sim_img[view] = sim_images[idx][k_dist[view]]
                                    sim_model[view], sim_id[view] = sim_img[view].split(
                                        '_')

                            sim_img_path = ["" for x in range(views - 1)]
                            for view in range(views - 1):
                                sim_img_path[view] = join(
                                    data_dir, categ, sim_model[view])
                            img_str = read_img(join(data_dir, categ, model),
                                               image_ids[idx], sim_img_path, sim_id)
                            writer.write(img_str)
                        elif save_pose:
                            img_str = read_pose(join(data_dir, categ, model),
                                                image_ids[idx])
                            writer.write(img_str)
                        elif save_pcl:
                            pcl_str = read_pcl(join(data_dir, categ, model))
                            writer.write(pcl_str)
                    except KeyboardInterrupt:
                        sys.exit()
                    except BaseException:
                        print idx
                        print model
                        N_missing += 1
                        continue

            if not load_filenames:
                np.save(names_file, all_models)
            if save_img:
                np.save('%s/rand_file_list_%dviews.npy' %
                        (out_dir, views), rand_file_list)
            if save_mask:
                np.save('%s/rand_file_list_mask_%dviews.npy' %
                        (out_dir, views), rand_file_list_mask)

            print 'Time: ', (time.time() - time_st) // 60
            print 'N_missing: ', N_missing
