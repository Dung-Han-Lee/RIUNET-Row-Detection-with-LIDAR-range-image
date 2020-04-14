from sys import argv
from math import pi, radians, degrees
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import sys
import errno
from glob import glob

sys.path.append('../')
from visualize import image_histogram_equalization, topview
from train import paths

VELODYNE_VERTICAL_RES = radians(2)
VELODYNE_VERTICAL_MIN_ANGLE = radians(-15)
VELODYNE_NUM_BEAMS = 16

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_rangeview(pointcloud, theta_min=-pi/2, theta_max=pi/2, fix_len=True):
    """
    Args:
        pointcloud: 
            Nx4 numpy array of 3D point coordinates in lidar sensor frame
            the corresponding fields are x,y,z,intensity
        theta_max/min:  
            the allowed horizontal theta range with forward defined as 0, 
            left as pi/2 (due to arctan2 property) 

    Returns:
        Range view image

    Notes: 
        z is align with rotation axis pointing outward from the top of device
        x pointing forward (due to ROS convention) 
    """
    # Convert cartesian to spherical
    r = np.linalg.norm(pointcloud[:,0:3], axis=1)
    phi = np.arccos(pointcloud[:,2] / r) - (pi / 2) # top: -pi/2, down: pi/2
    theta = np.arctan2(pointcloud[:,1], pointcloud[:,0])

    # Calculate vetical phi with top beam corrs to index 0, btm index 15
    y = np.rint((phi - VELODYNE_VERTICAL_MIN_ANGLE) * (1./VELODYNE_VERTICAL_RES)).astype('int')

    # Calculate horizontal theta with right corrs to index 0 (need flip later)
    theta_step = radians(0.35)
    x_len = 512 if fix_len else int((theta_max - theta_min) / theta_step )
    x = np.rint((theta - theta_min) * (1./theta_step)).astype('int')

    # Initialize output images
    img_shape = (VELODYNE_NUM_BEAMS, x_len)
    range_img = np.zeros(img_shape)
    value_img = np.zeros(img_shape)

    # Discard points outside the allowed horizontal theta range
    valid = (x >= 0) & (x < x_len)

    # Write values into output images
    r_valid = r[valid]
    x_valid = x[valid]
    y_valid = y[valid]
    range_img[y_valid, x_valid] = r_valid
    value_img[y_valid, x_valid] = pointcloud[valid,3]

    # correct the arctan2 issue
    range_img = np.fliplr(range_img) 
    value_img = np.fliplr(value_img)

    # Augment the image by repeating each beam FACTOR times
    FACTOR = 4
    aug_range_img = np.zeros((FACTOR*VELODYNE_NUM_BEAMS, x_len))
    aug_value_img = np.zeros((FACTOR*VELODYNE_NUM_BEAMS, x_len))
    for i in range(VELODYNE_NUM_BEAMS):
        aug_range_img[FACTOR*i:FACTOR*i + FACTOR, :] = range_img[i, :]
        aug_value_img[FACTOR*i:FACTOR*i + FACTOR, :] = value_img[i, :]

    return aug_value_img, aug_range_img

if __name__ == '__main__':

    # Top view (LIDAR coord) controller
    birdview = 0

    # Collect folder and npz file pairs
    npzs_dir = paths.data_dir
    output_dir = paths.data_dir + "range_view/"
    ext = ".npz"
    pairs = [(x[0], y) for x in os.walk(npzs_dir) \
                for y in glob(os.path.join(x[0], '*' + ext))]
    
    # Create range images from npz pointclouds
    mkdir_p(output_dir)
    for idx, (folder, npz) in enumerate(sorted(pairs, key = lambda x: x[1])):
        
        path = os.path.join(output_dir, folder[len(npzs_dir):])
        mkdir_p(path)

        stamp = npz.split("/")[-1][:-len(ext)]
        rname = stamp + "_range" 
        iname = stamp + "_intensity"
        vname = stamp + "_visualize.jpg"

        pc = np.load(npz)["pointcloud"]
        iimg, rimg = get_rangeview(pc)

        cv2.imwrite(os.path.join(path, vname) , 10.0 * rimg)

        for name, img in [(rname, rimg), (iname, iimg)]:
            np.save(os.path.join(path, name) , img)
        
        if birdview:
            cv2.imshow("topview", topview(pc))
            cv2.waitKey(0)
