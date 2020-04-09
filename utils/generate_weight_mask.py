import os
import cv2
import imutils
import numpy as np
from sys import argv

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