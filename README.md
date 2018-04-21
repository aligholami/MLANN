# MLANN

<p align="center">
    <img src="http://uupload.ir/files/665d_untitled.png">
</p>

<p align="center">
    <h4 align="center"> || Implementation of Maximum Likelihood Approximate Neareset Neighbour Alogrithm for Realtime Image Recognition || </h4>
</p>

---
#### Overview
I've used **face_recognition**, **OpenCV** and **dlib** library to implement this face recognizer. This is using the **Nearest Neighbor** method by default.

##### Frame Preprocessing
* Resize frame of video to 1/4 size for faster face recognition processing
* Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

##### Realtime Recognition
* Find all the faces and face encodings in the current frame of video
* Find the best match using **Nearest Neighbor** method

##### Feature Extraction
* Feature extraction with [Inception Resnet V2](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1) using Tensorflow