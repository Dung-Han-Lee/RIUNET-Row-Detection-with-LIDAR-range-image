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

from sys import argv
import os
import errno
from glob import glob
import rosbag
import numpy as np
from numpy_pc2 import pointcloud2_to_xyzi_array

pointcloud_topic = "/velodyne_points"

raw_dir = "../raw/"
bags_dir = argv[1] if len(argv) == 2 else "../bag"
ext = '.bag'

# Walk through all the sub-folder and collect bag names
bagnames = [y for x in os.walk(bags_dir) for y in glob(os.path.join(x[0], '*' + ext))]
filenames = [b[len(bags_dir):-len(ext)].replace("/","_") for b in bagnames]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

for filename, bagname in zip(filenames, bagnames):
    bag = rosbag.Bag(bagname)
    data = bag.read_messages(
        topics=[pointcloud_topic])

    count = 0
    for topic, msg, t in data:
        if topic == pointcloud_topic:
            pc = pointcloud2_to_xyzi_array(msg) 
            timestamp = msg.header.stamp.to_nsec()
            timestamp_arr = np.array([timestamp], dtype='int64')
            
            d = "%s%s" % (raw_dir, filename)
            mkdir_p(d)

            path =  "%s/%s" % (d, timestamp)

            np.savez(path, pointcloud=pc, timestamp=timestamp_arr)
