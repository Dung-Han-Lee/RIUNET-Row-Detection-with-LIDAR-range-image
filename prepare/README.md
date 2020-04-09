## Overview
The original dataset is not provided in this repo, but some example data  
are given to show structure and file types needed for this application.  

## Extract pointcloud from bag files
The following command would extract pointcloud, convert it to numpy and  
store it under RIUNET/raw for futher process
     python cvt_bag2npyz.py  ../bag

## Create range images
The following command would convert numpy pointcloud to range images and  
store them under RIUNET/raw/range_view
    python get_rangeview.py ../raw/
