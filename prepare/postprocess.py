import os
import cv2
import imutils
import numpy as np
import pdb
import errno
from sys import argv

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rotate(directory):
    angles = ([5, 10, 15, -5, -10, -15])
    for filename in sorted(os.listdir(directory)):
        ext = filename.split(".")[-1]
        ipath = os.path.join(directory, filename)
        if(ext != 'png' and ext != 'npy'):
            continue

        for angle in angles:
            opath = os.path.join(directory,
                                 filename.split(".")[0]+"_"+str("%02d"%angle)) 
            if ext == 'png': #labels 
                im = cv2.imread(ipath)   
                im = imutils.rotate(im,angle)
                im[im >= 128] = 255
                im[im <  128] = 0
                cv2.imwrite(opath +"." + ext, im)


            if ext == 'npy':
                im = np.load(ipath, allow_pickle=True)
                im = imutils.rotate(im,angle)
                np.save(opath, im)

def rename(directory):
    num = 0
    for filename in sorted(os.listdir(directory)):
        ext = filename.split(".")[-1]
        if (ext != 'png' and ext != 'npy' and ext != 'jpg') :
            continue
        os.rename( os.path.join(directory, filename),\
                os.path.join(directory, ("%04d" % num ) + "." + ext) )
        num +=1

def fix_color(image_directory):
    for filename in os.listdir(image_directory): #assuming png
        if (filename.split(".")[-1] != 'jpg') :
            continue 
        im = cv2.imread(os.path.join(image_directory,filename), 0)
        im[im > 128] = 255
        im[im < 128] = 0
        cv2.imwrite(os.path.join(image_directory,filename.split(".")[0] + '.png'), im)

def generate_weight_map(img, weight=[0.05, 0.95], border_cost=10.):
    """
    Args:
        annotation (H W 1) numpy gray scale image
        weights: [background, annotation]
    Return:
        weight_map (H W 1) numpy weighting for loss 
    """
    contours,_ = cv2.findContours((img > 0 ).astype('uint8'),
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_NONE)

    # Prepare augmentation on the border
    aug  = np.zeros_like(img).astype(np.float32)
    cv2.drawContours(aug, contours, -1,  border_cost,  1)
    kernel = np.array(\
            [[0. ,  0.,  0. , 0. , 0. ],
             [0. ,  0.,  0.3, 0. , 0. ],
             [0.1, 0.3,  1.0, 0.3, 0.1],
             [0. ,  0.,  0.3, 0. , 0. ],
             [0. ,  0.,  0. , 0. , 0. ]]).astype(np.float32)
    kernel = kernel/np.sum(kernel)
    aug = cv2.filter2D(aug,-1,kernel)

    # Augment the weighitng on the border
    out  = np.ones_like(img).astype(np.float32) * weight[0]
    cv2.drawContours(out, contours, -1, weight[1], -1)
    out[aug > 0] += aug[aug > 0]
    return out

    """
    Example usage:
    outdir = os.path.join(argv[1], "weights")
    mkdir_p(outdir)
    for f in sorted(os.listdir(argv[1])):
        if(f.split(".")[-1] != "png"):
            continue
        impath = (os.path.join(argv[1], f))
        img  = cv2.imread(impath, 0)
        wimg = generate_weight_map(img)
        ext = ".npy"
        outpath = os.path.join(outdir, f.split(".")[0] + ext)
        np.save(outpath , wimg)
    """

if __name__ == '__main__':
    rename(argv[1])

