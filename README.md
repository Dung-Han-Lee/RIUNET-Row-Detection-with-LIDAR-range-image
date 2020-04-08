## Row Detection for Agriculture Robot using RIU-Net
This repo contains python codes that detects traversible row for agriculture robots  
using [UNet](https://arxiv.org/abs/1505.04597). This work largely follows the paper  
[RIU-Net: Embarrassingly simple semantic segmenta-tion of 3D LiDAR point cloud](https://arxiv.org/abs/1905.08748)  
Note that the very same pipeline could be used for generic object detection, while this  
repo is only concerns about detecting a single row without row switching.

The purpose of this work is to enable autonomous, visual-based in-row navigation for  
agriculture robots. Specifcially, to address the occassions where GPS are not reliable  
 or not accurate enough. The training data were collected from a vineyard field using  
Velodyne's VLP-16, which was mounted on a mobile agriculture robot. 

## Quantitative Results
This work was implemented with 150 training images (augmented to 1050). As a result: with   
50 test images, an average of 78% intersection over union performance has been achieved.  
Also, 48 out of 50 tests were passed with the definition of success being:   
* the orientation of prediction-line and human-annotated-line is no different than 3 degrees 
* the RMS distance between two end-points of human-annotated-line and prediction-line   
  are less than 0.5 meters.

## Run with Visualization
  
  *single model
  
    python3 inference.py --m ./weight/unet.pth --show "all"
    
  *multiple model
  
    python3 inference.py --m ./weight/*.pth --show "all"
