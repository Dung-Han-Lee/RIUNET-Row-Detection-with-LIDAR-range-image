#
# Author : Dung-Han Lee
# contact: dunghanlee@gmail.com
# 
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import cv2
import numpy as np
import scipy.misc as m
import torch
import pdb
import argparse
import sys
from sys import exit
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LinearRegression

#User Define modules
sys.path.append('./train')
import models
import data
import paths
from visualize import testdemo, deproject_row_points

class Test:
    def __init__(self, path_weight):
        self.device = torch.device('cuda' if \
            torch.cuda.is_available() else 'cpu') 
        
        # Initialize model
        weight = torch.load(path_weight, map_location = self.device)
        self.network = models.unet()
        self.network.load_state_dict(weight)
        self.network.eval()
        self.network.to(self.device)

    def denoise(self, img):
        """
        Args: 
            numpy binary image (positive means traversible path)
        Returns:
            removed noise by keeping the max contour only*
            *assuming the traversible path is indeed the largest contour
        """

        contours,_ = cv2.findContours((img > 0 ).astype('uint8'),
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
        out = np.zeros_like(img).astype(np.uint8)
        idx = None
        amax = -1
        for i , ctr in enumerate(contours):
            if cv2.contourArea(ctr) > amax:
                amax = cv2.contourArea(ctr)
                idx = i
        if (idx==None):
            return img
        cv2.drawContours(out, contours, idx, 255, -1)
        return out

    def get_line(self, points):
        '''
        Args: 
            point: N X 3 numpy array representing pointcloud, 
                   each row is (x, y, z)
        Returns: 
            degree of the least-square-fitted line 
        '''
        lr = LinearRegression()
        xs = points[:, 0].reshape(-1, 1)
        ys = points[:, 1].reshape(-1, 1)
        lr.fit(xs, ys)

        xmin = np.min(xs)
        xmax = np.max(xs) 
        xs = np.array([xmin, xmax])
        ys = np.array([lr.predict([[xmin]]), lr.predict([[xmax]])])
        ys = ys.flatten()
        return xs, ys, lr.coef_

    def success(self, prediction, labels, rimg):
        """
        Args:
            prediction H X W numpy gray scale image, traversible is 255
            label      H X W numpy binary image
            rimg       H X W numpy gray scale range image
        Returns:
            whether predicted line is within error bound compared 
            to labeled line
        """
        pc_res = deproject_row_points(prediction, rimg.copy())
        pc_lbl = deproject_row_points(labels, rimg.copy())

        # two set of vertex points that define a line
        x_res, y_res, coef_res = self.get_line(pc_res)
        x_lbl, y_lbl, coef_lbl = self.get_line(pc_lbl)

        # compute orientation from gradient
        deg_res = np.rad2deg(np.arctan2(y_res[-1] - y_res[0], x_res[-1] - x_res[0]))
        deg_lbl = np.rad2deg(np.arctan2(y_lbl[-1] - y_lbl[0], x_lbl[-1] - x_lbl[0]))

        # compute distance base on labeled-line's two vertices
        y_res_dw = coef_res[0] * (x_lbl[0] - x_res[0]) + y_res[0]
        y_res_up = coef_res[0] * (x_lbl[1] - x_res[1]) + y_res[1]
        y_dist = (np.linalg.norm(np.array([y_res_dw, y_res_up]).flatten() -y_lbl)) 
        #print(y_dist)
        return abs(deg_res - deg_lbl) < 3.0 and y_dist <= 0.5

    def evaluate(self, test_loader, visualize_flag):

        res_iou = []
        res_align = []
        for num, (rimg, feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(self.device), labels.to(self.device)
            seg_im = self.network(feats)
            
            # Convert tensor to numpy
            to_numpy = lambda x : x[0].cpu().detach().numpy()
            (rimg, seg_im, labels) = map(to_numpy, (rimg, seg_im, labels))

            # Compute range images
            imax = seg_im.argmax(axis = 0)
            prediction = self.denoise(imax==1)

            # Compute iou
            intersec = ( labels > 0 ) & (prediction > 0)
            union    = ( labels > 0 ) | (prediction > 0) 
            iou = np.sum(intersec)/np.sum(union) 
            res_iou.append(iou)

            # Determine success of alignment between predict line & annotated line
            success = self.success(prediction, labels, rimg)
            res_align.append(success)

            # Visualize model output and deprojected pointcloud
            if(visualize_flag == "all"):
                #testdemo(prediction, labels, rimg, birdview=False)
                testdemo(prediction, labels, rimg, birdview=True)

            elif(visualize_flag == "failed" and not success):
                print("failed id = {}".format(num))
                testdemo(prediction, labels, rimg, birdview=False)
                testdemo(prediction, labels, rimg, birdview=True)

        #print(res_iou)
        print("iou = {:.3f}, success rate = {:.2f}".format(\
            np.mean(res_iou), np.mean(res_align)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evalutate results of range image unet.')
    parser.add_argument("-m", "--model", required=True, nargs="+",  help="paths to deep learning segmentation model")
    parser.add_argument("-s", "--show", choices = ["all", "failed", "none"], default="none",
	    help="whether or not to visualize results")
    args = vars(parser.parse_args())

    test_dataset   = data.RangeViewDataset(mode = 'test')
    test_loader = DataLoader(test_dataset)
    
    for path in (args["model"]):
        print(path)
        test = Test(path) #best 0406_RIU03_60
        test.evaluate(test_loader, args["show"])
