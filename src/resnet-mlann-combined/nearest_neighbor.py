import numpy as np
import sys
from config import *

class NearestNeighbor:

    def __init__(self):
        pass
    
    def brute_force(self, reference_faces, input_face):

        smallest_distance = sys.maxsize

        for ref_face in reference_faces:
            distance = np.sqrt(np.sum(np.square(
                ref_face[0:127] - input_face[0:127]
            )))

            if(distance < smallest_distance):
                smallest_distance = distance

        return smallest_distance