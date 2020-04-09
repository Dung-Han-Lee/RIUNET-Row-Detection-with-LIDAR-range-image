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
import imutils
import numpy as np
from sys import argv

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

if __name__ == '__main__':
    rename(argv[1])

