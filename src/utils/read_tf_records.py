##############################################################################
# Extract data from tensorflow records file
# Here, data is a a set of images/pcl/pose values and corresponding filenames
# Verify that the data saved using create_tf_records.py is correct.
# Check the webpage below for more info
# https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
##############################################################################

import os, sys
from os.path import join, abspath, basename
import glob
import time

import tensorflow as tf
import numpy as np
import scipy.misc as sc

from helper_funcs import create_folder

data_dir = '../../data/ShapeNet_rendered'
data_dir_pcl = '../../data/ShapeNet_v1'
categs = ['02691156', '02958343', '03001627']

out_data_dir = '../../data/train_images'

batch_size = 1
verify = True
save_images = False

def extract_fn(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    image = tf.image.decode_png(sample['image'], 3, tf.uint8)
    name = sample['filename']
    return image, name


def extract_fn_mask(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    image = tf.image.decode_png(sample['image'], 1, tf.uint8)
    name = sample['filename']
    return image, name


def extract_fn_2(data_record, type='png'):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'filename_2': tf.FixedLenFeature([], tf.string),
        'image_2': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    if type=='jpg':
        image = tf.image.decode_jpeg(sample['image'], 3, tf.float32)
    else:
        image = tf.image.decode_png(sample['image'], 3, tf.uint8)
        image_2 = tf.image.decode_png(sample['image_2'], 3, tf.uint8)
    name = sample['filename']
    name_2 = sample['filename_2']
    return image, name, image_2, name_2


def extract_fn_pose(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'pose': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    pose = sample['pose']
    pose = tf.decode_raw(pose, tf.float64)
    name = sample['filename']
    return pose, name


def extract_fn_pcl(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'pcl': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    pcl = sample['pcl']
    pcl = tf.decode_raw(pcl, tf.float64)
    name = sample['filename']
    return pcl, name

modes = ['test']
#modes = ['train', 'val', 'test']

for mode in modes:
    for categ in categs:
        read_img = False
        read_mask = False
        read_pose = False
        read_pcl = True
        if read_img:
            records_file = abspath('../../data/%s_%s_image.tfrecords'%(categ, mode))
        elif read_mask:
            records_file = abspath('../../data/%s_%s_mask.tfrecords'%(categ, mode))
        elif read_pcl:
            records_file = abspath('../../data/%s_%s_pcl_color.tfrecords'%(categ, mode))
        elif read_pose:
            records_file = abspath('../../data/%s_%s_pose.tfrecords'%(categ, mode))


        models = sorted(os.listdir(join(data_dir, categ)))

        dataset = tf.contrib.data.TFRecordDataset([records_file])
        if read_img:
            dataset = dataset.map(extract_fn, num_threads=6)
        if read_mask:
            dataset = dataset.map(extract_fn_mask, num_threads=6)
        elif read_pose:
            dataset = dataset.map(extract_fn_pose, num_threads=6)
        elif read_pcl:
            dataset = dataset.map(extract_fn_pcl, num_threads=6)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        next_element = iterator.get_next()

        _data = []; images = []; names = [];
        diff = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            time_st = time.time()
            try:
                out_dir = join(out_data_dir, categ, mode)
                create_folder([out_dir])
                while True:
                    _img, _img_name = sess.run(next_element)
                    images.append(_img)
                    names.append(_img_name)
                    if save_images:
                        sc.imsave(join(out_dir, _img_name[0]+'.png'),
                                _img[0])
            except KeyboardInterrupt:
                sys.exit()
            except:
                # Verify that the images were written and read correctly
                if verify:
                    for idx, name in enumerate(names):
                        if idx%1000==0:
                            print idx, '/', len(names)
                            print 'diff: ', diff
                        if read_img:
                            pdb.set_trace()
                            model_name, model_num = name[0].split('_')
                            img = sc.imread(join(data_dir,categ,model_name,'render_%s.png'%model_num))
                            diff += np.mean(np.abs(img[:,:,:3]-images[idx][0]))
                        elif read_mask:
                            model_name, model_num = name[0].split('_')
                            img = sc.imread(join(data_dir,categ,model_name,'mask_%s.png'%model_num))
                            diff += np.mean(np.abs(img-images[idx][0,:,:,0]))
                        elif read_pose:
                            model_name, model_num = name[0].split('_')
                            with open(join(data_dir,categ,model_name,'view.txt')) as fp:
                                angles = [item.split('\n')[0] for item in
                                    fp.readlines()]
                            angle = angles[int(model_num)]
                            angle = [float(item)*(np.pi/180.) for item in angle.split(\
                                    ' ')[:2]]
                            diff += np.mean(np.abs(angle-images[idx]))
                        elif read_pcl:
                            pdb.set_trace()
                            model_name = name[0].split('_')[0]
                            pcl = np.load(join(data_dir_pcl,categ,model_name,'pcl_1024_fps_trimesh_colors.npy'))
                            diff += np.mean(np.abs(pcl-np.reshape(images[idx],
                                (1024,6))))

                    print 'Error: ', diff

                time_end = time.time()
                print 'Time: ', time_end - time_st
                print 'Done!'
