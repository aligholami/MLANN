# MLANN

<p align="center">
    <img src="http://uupload.ir/files/665d_untitled.png">
</p>

<p align="center">
    <h4 align="center"> || Implementation of Maximum Likelihood Approximate Neareset Neighbour Alogrithm for Realtime Image Recognition || </h4>
</p>

---
#### Overview


#### Step-by-step Guidline
*   Extract frames from video capture.
*   Detect face regions.
*   Crop faces and align them.
*   Each face is assigned to one of these categories; **right**, **left**, **center**. (Pose Invariant Recognition)
*   Extract feature maps of shape (128, 1) from images.
*   Perform an initial search among categories.
*   Perform an exhaustive search of all dataset using **ANN**(Approximate Nearest Neighbor) method.

