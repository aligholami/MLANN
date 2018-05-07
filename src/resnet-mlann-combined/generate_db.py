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

# Some scaling configs
RESCALE_FACTOR = 2



# Create Tensorflow face recognition graph
FRGraph = FaceRecGraph()

# Create face detection objects 
face_detector = MTCNNDetect(FRGraph, scale_factor=RESCALE_FACTOR)

# Open up the faces database(read permission only)
f = open('./faces_db.txt', 'r')

# Load the database into the ram
data_set = json.loads(f.read())
