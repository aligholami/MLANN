# ========================================
# [] File Name : main.py
#
# [] Creation Date : May 2018
#
# [] Author 2 : Ali Gholami
#
# ========================================

import numpy as np
from mtcnn.mtcnn import MTCNN
from mtcnn_detect import MTCNNDetect
from align_custom import AlignCustom
from face_feature import FaceFeature
from tf_graph import FaceRecGraph
import json
from PIL import Image
import glob
import os
import cv2
import path


RESCALE_FACTOR = 2  # Some scaling configs
MIN_FACE_SIZE = 150  # Minimum size of 80*80 for each face
DESIRED_SIZE = 160  # Desired size after the image alignment



# Create Tensorflow face recognition graph
FRGraph = FaceRecGraph()

# Create the aligner instance
aligner = AlignCustom()

# Create the face feature extractor object
feature_extractor = FaceFeature(FRGraph)

# Create face detection objects 
# face_detector = MTCNNDetect(FRGraph, scale_factor=RESCALE_FACTOR)
face_detector = MTCNN()

# Open up the faces database(read permission only)
f = open('./faces_db.txt', 'r')

# Load the database into the ram
data_set = json.loads(f.read())

# Dynamic batch folder selector
ROOT_DIRECTORY = './lfw/'

for subdir, dirs, files in os.walk(ROOT_DIRECTORY):
    for image in files:

        # Convert the image into the proper format
        img_name = os.path.basename(subdir).replace("_", " ")
        
        image_path = subdir + "/" + image
        image = cv2.imread(image_path)

        

        detected = face_detector.detect_faces(image)

        if(len(detected) != 0):
            detected = detected[0]

            rects, landmarks = detected["box"], detected["keypoints"]


            landmark_s = []
            # Configure landmarks -> Put them in a list
            for key, value in landmarks.items():
                for val in value:            
                    landmark_s.append(val)

            print("[INFO] Landmarks are ", landmark_s)

            person_imgs_from_different_angles = {"Left" : [], "Right": [], "Center": []}
            person_features_from_different_angles = {"Left" : [], "Right": [], "Center": []}

            # Iterate through all rects in that image(there will be one in this case)
            # Align image and find the position of the face
            aligned_image, pos = aligner.align(DESIRED_SIZE, image, landmark_s);
            
            if len(aligned_image) == DESIRED_SIZE and len(aligned_image[0]) == DESIRED_SIZE:   
                # Load the aligned face to the proper angle
                person_imgs_from_different_angles[pos].append(aligned_image)



            print("[INFO] Number of persons in the image: ", len(rects)/4)

            # Extract the features from images
            for pos in person_imgs_from_different_angles:
                if (pos == "Center"):
                    try:
                        person_features_from_different_angles[pos] = [np.mean(feature_extractor.get_features(person_imgs_from_different_angles[pos]), axis = 0).tolist()]
                    except Exception:
                        pass
                else:
                    person_features_from_different_angles[pos] = [[0 for i in range(128)]]
            
            data_set[img_name] = person_features_from_different_angles

            # Write back to db
            f = open('./faces_db.txt', 'w')
            f.write(json.dumps(data_set))

            print("[INFO] Added image to database... ")

        else:
            pass
