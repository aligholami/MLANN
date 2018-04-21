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
        raise ValueError("Unimplemented mode")

def begin_camera_session():
    print("[INFO] Initialized camera session...")

    # Get the webcam handle
    vs = cv2.VideoCapture(0)

    while True:
        # Capture frame by frame
        _,frame = vs.read();

        # Find the faces in the frame
        rects, landmarks = face_detect.detect_face(frame, MIN_FACE_SIZE)

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
                print("Face alignment failed.")      

        # In case any faces found properly
        if(len(aligns) > 0):

            # An array of feature maps
            feature_mmap = extract_feature.get_features(aligns)

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

    f = open('./facerec_128D.txt', 'r')

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


'''
Description:
User input his/her name or ID -> Images from Video Capture -> detect the face -> crop the face and align it 
    -> face is then categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Append each newly extracted face 128D vector to its corresponding position type (Center, Left, Right)
    -> Press Q to stop capturing
    -> Find the center ( the mean) of those 128D vectors in each category. ( np.mean(...) )
    -> Save
    
'''
def add_new_user():
    vs = cv2.VideoCapture(0); #get input from webcam
    
    ("Please input new user ID:")
    new_name = input(); #ez python input()
    f = open('./facerec_128D.txt','r');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main(args);
