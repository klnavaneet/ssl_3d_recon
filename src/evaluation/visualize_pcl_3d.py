###############################################################################
# Visualize the gt and reconstructed 3D point clouds from saved points clouds
# from directory
###############################################################################

import os, sys
from os.path import join, abspath
import time
import pdb
import glob

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as sc
from multiprocessing import Pool
import cv2

sys.path.append('../utils')
sys.path.append('../chamfer_utils')
import show3d_balls
from helper_funcs import create_folder, remove_outliers
from shapenet_taxonomy import shapenet_category_to_id

exp_dir = '../../expts/temp'
log_dir = join(exp_dir, 'log_proj_pcl_test')
categ = 'chair'
categ = shapenet_category_to_id[categ]
pcl_data_dir = '../../data/ShapeNet_v1/%s'%(categ)
data_dir = '../../data/ShapeNet_rendered/%s'%(categ)
mode = 'test'
models = sorted(np.load('../../splits/images_list_%s_%s.npy'%(categ, mode)))

names = sorted(glob.glob(join(log_dir, '*.npy')))

# Number of outputs to visualize
n_plots = 100

def viz_pcl(ballradius=3):
    '''
    Visualize the input image, GT and predicted point cloud
    '''
    for idx in range(n_plots):
        img_name, img_id = models[idx][0].split('_')

        # Load the gt and pred point clouds
        gt_pcl = np.load(join(pcl_data_dir, img_name,
                'pcl_1024_fps_trimesh.npy'))
        pcl = np.load(names[idx])[:,:3]
        pcl = remove_outliers(pcl)

        # Load and display input image
        ip_img = sc.imread(join(data_dir,
            img_name,'render_%s.png'%(img_id)))

        # RGB to BGR for cv2 display
        ip_img = np.flip(ip_img[:,:,:3], -1)
        cv2.imshow('', ip_img)

        show3d_balls.showpoints(gt_pcl, ballradius=ballradius)
        show3d_balls.showpoints(pcl, ballradius=ballradius)
        saveBool = show3d_balls.showtwopoints(gt_pcl, pcl, ballradius=ballradius)


viz_pcl()
