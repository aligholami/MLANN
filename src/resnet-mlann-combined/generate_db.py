# ========================================
# [] File Name : main.py
#
# [] Creation Date : May 2018
#
# [] Author 2 : Ali Gholami
#
# ========================================

from mtcnn_detect import MTCNNDetect
from align_custom import AlignCustom
from face_feature import FaceFeature
from tf_graph import FaceRecGraph
import json
from PIL import Image
import glob
import os
import path


RESCALE_FACTOR = 2  # Some scaling configs
MIN_FACE_SIZE = 80  # Minimum size of 80*80 for each face
DESIRED_SIZE = 160  # Desired size after the image alignment



# # Create Tensorflow face recognition graph
# FRGraph = FaceRecGraph()

# # Create face detection objects 
# face_detector = MTCNNDetect(FRGraph, scale_factor=RESCALE_FACTOR)

# # Create the face feature extractor object
# feature_extractor = FaceFeature(FRGraph)

# # Open up the faces database(read permission only)
# f = open('./faces_db.txt', 'r')

# # Load the database into the ram
# data_set = json.loads(f.read())

# Dynamic batch folder selector
ROOT_DIRECTORY = './lfw/'

for subdir, dirs, files in os.walk(ROOT_DIRECTORY):
    for file in files:

        # Convert the image into the proper format
        img_name = os.path.basename(subdir).replace("_", " ")
        



# while True:

    # # Load a batch of images from same person into the memory
    # image_batch = {}

    # # Extract the features of every image in the batch
    # for image in image_batch.items():
    #     rects, landmarks = face_detector.detect_face(image, MIN_FACE_SIZE)
        

    #     person_imgs_from_different_angles = {"Left" : [], "Right": [], "Center": []}
    #     person_features_from_different_angles = {"Left" : [], "Right": [], "Center": []}

    #     # Iterate through all rects in that image(there will be one in this case)
    #     for(i, rect) in enumerate(rects):

    #         # Align image and find the position of the face
    #         aligned_image, pos = aligner.align(DESIRED_SIZE, image, landmarks[i]);
            
    #         if len(aligned_image) == DESIRED_SIZE and len(aligned_image[0]) == DESIRED_SIZE:
            
    #         # Load the aligned face to the proper angle
    #         person_imgs_from_different_angles[pos].append(aligned_frame)

    #     # Extract the features from images
    #     for pos in person_imgs_from_different_angles:
    #         person_features_from_different_angles[pos] = [np.mean(feature_extractor.get_features(person_imgs_from_different_angles[pos]), axis = 0).tolist()]
        
    #     data_set[person_name] = person_features_from_different_angles

    #     # Write back to db
    #     f = open('./faces_db.txt', 'w')
    #     f.write(json.dumps(data_set))


