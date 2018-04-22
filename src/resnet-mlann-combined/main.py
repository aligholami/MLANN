# ========================================
# [] File Name : main.py
#
# [] Creation Date : April 2018
#
# [] Author 1 : David Vu
# [] Author 2 : Ali Gholami
#
# ========================================

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np


MIN_FACE_SIZE = 80  # Minimum size of 80*80 for each face
DESIRED_SIZE = 160  # Desired size after the image alignment

def main(args):
    mode = args.mode

    if(mode == "camera"):
        begin_camera_session()
    elif mode == "input":
        add_new_user()
    else:
        raise ValueError("\n [ERROR] Unimplemented mode")

def begin_camera_session():
    print("\n [INFO] Initialized camera session...")

    # Get the webcam handle
    vs = cv2.VideoCapture(0)

    while True:
        # Capture frame by frame
        _,frame = vs.read();

        # Find the faces in the frame
        rects, landmarks = face_detector.detect_face(frame, MIN_FACE_SIZE)

        aligns = []
        positions = []

        # Iterate the found faces
        for (i, rect) in enumerate(rects):

            # Align the faces
            aligned_face, face_pos = aligner.align(DESIRED_SIZE, frame, landmarks[i])

            if len(aligned_face) == DESIRED_SIZE and len(aligned_face[0]) == DESIRED_SIZE:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("\n [INFO] Face alignment failed.")      

        # In case any faces found properly
        if(len(aligns) > 0):

            # An array of feature maps
            feature_mmap = feature_extractor.get_features(aligns)

            # Find known faces
            recognized_faces = find_known_faces(feature_mmap, positions);

            # Display information for all of the rects
            for (i, rect) in enumerate(rects):

                # Rename coordinations
                x, y, width, height = rect[0], rect[1], rect[2], rect[3]

                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0))

                # Rename recognized_faces information
                face_name = recognized_faces[i][0]
                recognition_confidence = recognized_faces[i][1]

                # Display information below the bounding box
                cv2.putText(frame, face_name + " - " + str(recognition_confidence) + "%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

        # Display the new frame
        cv2.imshow("Frame",frame)

        # Exit on interrupt
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''
def find_known_faces(feature_mmap, positions, threshold = 0.6, p_threshold = 70):
    '''
        :param feature_mmap: an array of feature maps with shape (128, 1)
        :param positions: a list of face position types of all faces on screen
        :param thres: distance threshold
        :return: face name and confidence percentage
    '''

    f = open('./faces_db.txt', 'r')

    data_set = json.loads(f.read())
    known_faces_info = []

    for (i, input_feature_map) in enumerate(feature_mmap):
        result = "Unknown"
        smallest = sys.maxsize

        for person in data_set.keys():
            person_data = data_set[person][positions[i]]

            # One of the checkpoints of this project
            for known_feature_map in person_data:
                distance = np.sqrt(np.sum(np.square(known_feature_map - input_feature_map)))

                if(distance < smallest):
                    smallest = distance
                    result = person

        percentage =  min(100, 100 * threshold / smallest)

        # In case the confidence percentage is not satisfying
        if percentage <= p_threshold:
            result = "Unknown"

        known_faces_info.append((result, percentage))

    return known_faces_info

def add_new_user():
    
    # Get the webcam handle
    vs = cv2.VideoCapture(0)
    
    print("\n [INFO] Welcome to BioFace.")
    print("\n What's your name?")
    new_name = input()

    # Open up the faces database(read permission only)
    f = open('./faces_db.txt', 'r')

    # Load the database into the ram
    data_set = json.loads(f.read())


    person_imgs = {"Left" : [], "Right": [], "Center": []}
    person_features = {"Left" : [], "Right": [], "Center": []}

    # Some acknowledgements
    print("\n [INFO] Please start turning slowly...")
    print("\n [INFO] Press q to save and exit.")

    while True:

        # Capture frame by frame
        _, frame = vs.read();
        rects, landmarks = face_detector.detect_face(frame, MIN_FACE_SIZE)

        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(DESIRED_SIZE, frame, landmarks[i]);
            if len(aligned_frame) == DESIRED_SIZE and len(aligned_frame[0]) == DESIRED_SIZE:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)

        # Exit on interrupt     
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Save face features to the dataset
    for pos in person_imgs:
        person_features[pos] = [np.mean(feature_extractor.get_features(person_imgs[pos]), axis=0).tolist()]
    data_set[new_name] = person_features;

    # Open up the faces database (Write permission only)
    f = open('./faces_db.txt', 'w');

    # Write and close the db
    f.write(json.dumps(data_set))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")

    args = parser.parse_args(sys.argv[1:])

    FRGraph = FaceRecGraph()

    aligner = AlignCustom()

    feature_extractor = FaceFeature(FRGraph)

    # Rescale for faster detection
    face_detector = MTCNNDetect(FRGraph, scale_factor=2)

    main(args);
