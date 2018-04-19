# ========================================
# [] File Name : knn.py
#
# [] Creation Date : April 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

import cv2
import face_recognition

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Extract the features of two image's
ali_gholami = face_recognition.load_image_file("ali.jpg")
obama_face_encoding = face_recognition.face_encodings(ali_gholami)[0]

mohammad_khajavi = face_recognition.load_image_file("mohammad.jpg")
biden_face_encoding = face_recognition.face_encodings(mohammad_khajavi)[0]
