import os
import cv2
import errno
import numpy as np
import matplotlib.pyplot as plt
from math import pi, radians, degrees, cos, sin
from sys import argv, exit
from sklearn.linear_model import LinearRegression
import pdb

VELODYNE_NUM_BEAMS = 16
VELODYNE_VERTICAL_MAX_ANGLE = radians(15)
DEMO_IMAGE_WIDTH = 500
DEMO_IMAGE_HEIGHT = 500

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def image_histogram_equalization(image, number_bins=1024):
    # from http://www.janeriksolem.net/yrange9/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def topview(pointcloud, scale=10, bin=True):
    """
    Args:
        pointcloud: 
            N x D (D>=3), numpy array of 3D point coordinates in lidar sensor frame
            the corresponding fields are x,y,z,intensity etc
    Returns:
        image with following correspondance:
            x (forward in sensor frame) -> row
            y (horizon in sensor frame) -> col
    """
    row_range = DEMO_IMAGE_HEIGHT
    col_range = DEMO_IMAGE_WIDTH

    img = np.zeros((row_range,col_range)).astype(np.uint8)
    (x, y, z) = (pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])

    # Scale values (c=0 corres to center line)
    c = scale*y + col_range/2
    r = x*scale
    z = 255 * ((z - min(z))/(max(z) - min(z)))

    # Mask out invalid points
    valid = (r >= 0) & (r < row_range-1) &\
            (c >= 0) &  (c < col_range-1)
    r = np.rint(r).astype('int')
    c = np.rint(c).astype('int')
    z = np.rint(z).astype('int')
    rvalid = r[valid]
    yvalid = c[valid]
    zvalid = 1 if bin is True else z[valid]
    img[rvalid, yvalid] = zvalid

    # Flip the image so r=0 is at btm, c=0 at right
    img = np.fliplr(img)
    img = np.flipud(img)

    return img

def deproject_row_points(label, rimg, theta_step=0.35):
    """
    Args:
        label:(H x W) numpy semantic annotation of target 
        rimg: (H x W) numpy range image with intensity corresponds to range 
        theta step: resolution in theta (horizontal)
    Returns:
        pointcloud (N X 3) corresponding to label pixels (x, y, z)
    """

    # Segment ROI
    rimg[ label == 0 ] = 0
    
    h, w = rimg.shape
    compressed = np.zeros((int(h/4), w))
    for i in range(h):
        if(i%4==0):
            compressed[int(i/4)] = rimg[i]
    rimg = compressed
    h, w = compressed.shape

    # Map to (x,y,z) with r, theta, phi
    out = []
    for row in range(h):    # [0, 16] --> [15, -15]
        phi = VELODYNE_VERTICAL_MAX_ANGLE - radians(2 * row) 
        for col in range(w):    #[0, 511] --> [-pi/2, pi/2]
            theta = radians((w/2 - col)*theta_step)
            r = rimg[row][col]
            if r > 0: 
                z = r * sin(phi)
                r = r * cos(phi)
                x = r * cos(theta)
                y = r * sin(theta)
                out.append([x, y, z])

    return np.vstack(out)

def diff(res, lbl):
    """
    Supporting function to help visualization in testdemo
    """

    (res, lbl) = map(lambda x : x.astype(np.bool), (res, lbl))
    err = np.zeros_like(res).astype(np.uint8)
    err[(res ^ (res & lbl))] = 150
    err[(lbl ^ (res & lbl))] = 220
    err[0][0]  = 255 # anchor value for viz
    return err

def plot_centerline(idx, pointclouds, zoom):
    '''
    Args:
        drawing idx [0, 1, 2, 3] -> predict, label, err_img, src_img
        segmented pointclouds of row (N X 3)
        scaling factor 
    '''
    if idx > 2 or pointclouds is None:
        return

    if idx == 2:
        plot_centerline(0, pointclouds, zoom)
        plot_centerline(1, pointclouds, zoom)
        return

    lr = LinearRegression()
    points = pointclouds[idx]
    xs = points[:, 0].reshape(-1, 1)
    ys = points[:, 1].reshape(-1, 1)
    lr.fit(xs, ys)
    
    # Select two points to draw a line
    xmin = np.min(xs)
    xmax = np.max(xs) 
    xs = np.array([xmin, xmax])
    ys = np.array([lr.predict([[xmin]]), lr.predict([[xmax]])])
    ys = ys.flatten()

    # Coordinate transformation from sensor frame to image frame
    # sensor frame: x upward, y leftward
    # image  frame: x rightward, y downward
    xi = DEMO_IMAGE_WIDTH/2 - zoom * ys 
    yi = DEMO_IMAGE_HEIGHT  - zoom * xs 

    color = 'r' if idx == 0 else 'w'
    plt.plot(xi, yi, color)

def testdemo(prediction, labels, rimg, birdview=False):
    """
    Args:
        prediction H X W numpy gray scale image, traversible is 255
        label      H X W numpy binary image
        rimg       H X W numpy gray scale range image
    Returns:
        2 x 2 visualization board
        Notice: 
            birdview is w.r.t sensor frame i.e. looking down at 
            the top of LIDAR 
    """

    src = res = lbl = err = pointclouds = zoom = None
    if birdview == True : 
        zoom = 35 # larger is closer

        # Get pointcloud corresponds to binary mask
        pc_src = deproject_row_points(np.ones_like(rimg), rimg.copy())
        pc_res = deproject_row_points(prediction, rimg.copy())
        pc_lbl = deproject_row_points(labels, rimg.copy())
        pointclouds = [pc_res, pc_lbl]

        # Get topview(sensor frame) of pointcloud
        src = topview(pc_src, zoom)
        res = topview(pc_res, zoom)
        lbl = topview(pc_lbl, zoom)
        err = diff(res, lbl)

        # Dilate images for better visualization
        kernel = np.ones((3,3), np.uint8)
        (src, res, lbl, err) = map(lambda x : cv2.dilate(x, kernel), (src, res, lbl, err))
  
    else:
        src = image_histogram_equalization(rimg)
        res = prediction ; 
        lbl = labels.astype(np.uint) * 255
        err = diff(res, lbl)

    fig=plt.figure(figsize=(50, 50))
    names = ["output", "annotation", "error img", "input"]
    for i, img in enumerate([res, lbl, err, src]):
        fig.add_subplot(1, 4, i+1)
        plot_centerline(i, pointclouds, zoom)
        plt.imshow(img)
        plt.title(names[i])
    plt.show()



if __name__ == '__main__':
    test_deproj(argv[1], argv[2])
        