#################################################################
# Read data from tensorflow records file
#################################################################

import os, sys
from os.path import join, abspath, basename
import glob
import time
import pdb

import numpy as np
import scipy.misc as sc
import tensorflow as tf


# Load data using tfrecords - single image (no NN)
def extract_fn(data_record, type='png'):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    if type=='jpg':
        image = tf.image.decode_jpeg(sample['image'], 3, tf.float32)
    else:
        image = tf.image.decode_png(sample['image'], 3, tf.uint8)
    name = sample['filename']
    return image, name


# Load data using tfrecords - two image inputs (1 NN)
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


def extract_fn_partseg(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    image = tf.decode_raw(sample['image'], tf.uint8)
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


def extract_fn_mask_2(data_record):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'filename_2': tf.FixedLenFeature([], tf.string),
        'image_2': tf.FixedLenFeature([], tf.string)
        }
    sample = tf.parse_single_example(data_record, features)
    image = tf.image.decode_png(sample['image'], 1, tf.uint8)
    name = sample['filename']
    image_2 = tf.image.decode_png(sample['image_2'], 1, tf.uint8)
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


def fetch_data(records_file, batch_size, data_len, dtype=['rgb'], shuffle=True,
        num_threads=6):
    '''
    dtype is a list of kinds of data (e.g: ['mask', 'rgb']) that needs to be
    returned in every batch.
    '''
    datasets = []
    for i, name in enumerate(dtype):
        datasets.append(tf.contrib.data.TFRecordDataset([records_file[name]]))

    for i, dset in enumerate(datasets):
        if dtype[i]=='rgb':
            datasets[i] = datasets[i].map(extract_fn, num_threads=num_threads)
        if dtype[i]=='rgb_2':
            datasets[i] = datasets[i].map(extract_fn_2, num_threads=num_threads)
        elif dtype[i]=='mask':
            datasets[i] = datasets[i].map(extract_fn_mask, num_threads=6)
        elif dtype[i]=='mask_2':
            datasets[i] = datasets[i].map(extract_fn_mask_2, num_threads=6)
        elif dtype[i]=='partseg':
            datasets[i] = datasets[i].map(extract_fn_partseg, num_threads=6)
        elif dtype[i]=='pcl':
            datasets[i] = datasets[i].map(extract_fn_pcl, num_threads=6)
        elif dtype[i]=='pose':
            datasets[i] = datasets[i].map(extract_fn_pose, num_threads=6)

    dataset = tf.contrib.data.TFRecordDataset.zip(tuple(datasets)).repeat(1000)
    if shuffle:
	dataset = dataset.shuffle(data_len)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


# Fetch all data - mask, rgb, segmentation
def fetch_data_all(records_file, batch_size, data_len):
    dataset_rgb = tf.contrib.data.TFRecordDataset([records_file['rgb']])
    dataset_rgb = dataset_rgb.map(extract_fn, num_threads=6)

    dataset_mask = tf.contrib.data.TFRecordDataset([records_file['mask']])
    dataset_mask = dataset_mask.map(extract_fn_mask, num_threads=6)

    dataset_partseg = tf.contrib.data.TFRecordDataset([records_file['partseg']])
    dataset_partseg = dataset_partseg.map(extract_fn_partseg, num_threads=6)

    dataset = tf.contrib.data.TFRecordDataset.zip((dataset_rgb, dataset_mask,
        dataset_partseg))

    dataset = dataset.shuffle(data_len)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element
