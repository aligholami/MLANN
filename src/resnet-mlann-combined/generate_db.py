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
from tf_graph import FaceRecGraph
import json


RESCALE_FACTOR = 2  # Some scaling configs
MIN_FACE_SIZE = 80  # Minimum size of 80*80 for each face
DESIRED_SIZE = 160  # Desired size after the image alignment



# Create Tensorflow face recognition graph
FRGraph = FaceRecGraph()

# Create face detection objects 
face_detector = MTCNNDetect(FRGraph, scale_factor=RESCALE_FACTOR)

# Open up the faces database(read permission only)
f = open('./faces_db.txt', 'r')

# Load the database into the ram
data_set = json.loads(f.read())


while True:

    # Load a batch of images 
    image_batch = {}

    # Extract the features of every image in the batch
    for image in image_batch.items():
        rects, landmarks = face_detector.detect_face(image, MIN_FACE_SIZE)
        

        person_imgs = {"Left" : [], "Right": [], "Center": []}
        person_features = {"Left" : [], "Right": [], "Center": []}

        # Iterate through all rects in that image
        for(i, rect) in enumerate(rects):

            # Align image and find the position of the face
            aligned_image, pos = aligner.align(DESIRED_SIZE, image, landmarks[i]);
            
            if len(aligned_image) == DESIRED_SIZE and len(aligned_image[0]) == DESIRED_SIZE:
            person_imgs[pos].append(aligned_frame)
