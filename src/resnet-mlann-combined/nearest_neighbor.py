import numpy as np
import sys
from config import *

class NearestNeighbor:

    def __init__(self):
        pass
    
    def brute_force(self, reference_faces, position, input_face, input_face_positions):

        smallest_distance = sys.maxsize
        face_ID = "Unkown"

        for reference_face in reference_faces.keys():
            reference_face_data = reference_faces[reference_face][input_face_positions[position]]

            for i in reference_face_data:
                distance = np.sqrt(np.sum(np.square(
                    i[0:127] - input_face[0:127]
                )))

                if(distance < smallest_distance):
                    smallest_distance = distance
                    face_ID = reference_face

        return smallest_distance, face_ID