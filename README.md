# Pitting localization and segmentation using RCNN
In this work, we try to detect and localize surface defects of type "pitting" on Ball Screw Drive dataset [[1]](#1). The dataset consists of 1104  RGB high resolution images of which 394 are annotated with pitting defects, the pitting size varries between very small pittings and large ones. also the dataset contain images with multiple pittings having different sizes.

<p align="center">
  <img src="./images/dataset/1.png" width="450">
  <img src="./images/dataset/2.png"  width="350">
  <img src="./images/dataset/3.png" width="450">
  <img src="./images/dataset/4.png" width="350">
</p>

# model and training
- First we transform the data to COCO format ( the function is availble in `data.py` file ) and split it to training and testing set using cocosplit library available at the following link https://github.com/akarazniewicz/cocosplit.git.
- For the second step we construct our masked RCNN using pytorch, in our case we have 2 classes pitting and background (always present), the model construction is available int the file `model.py`. The model behave differently during training and inference. During training the  model return the value of RCNN 5 losses, while during testing it returns the set of boxes with their calssification and scores along with the binary masks for segmentation.
- Finally we train the model in 2 steps using dynamic wheighting of the losses such that the model correctly detect and predict the bounding boxes with high confidence first then focuses on the segmentation task. So for the first step we have high wheights for all the losses except the mask loss while in the second step all losses have approximately simillar wheights.

# prediction visualizations
<h3 align="center">real vs prediction</h3>
<p align="center">
  <img src="./images/predections/1_real.png" width="400">
  <img src="./images/predections/1_predection.png"  width="400">
  <img src="./images/predections/2_real.png" width="400">
  <img src="./images/predections/2_predection.png" width="400">
  <img src="./images/predections/3_real.png" width="400">
  <img src="./images/predections/3_predection.png" width="400">
</p>

# References
<a id="1">[1]</a>  Tobias Schlagenhauf, Magnus Landwehr, and Jürgen Fleischer: Industrial Machine Tool Component Surface Defect Dataset (2021). https://github.com/2Obe/BSData.git
