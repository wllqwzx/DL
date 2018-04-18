'''
Ref: http://dlib.net/face_landmark_detection.py.html
'''

import dlib
import glob
import os
from skimage import io

# pretrained predictor model path
predictor_path = './bin/shape_predictor_68_face_landmarks.dat'

faces_folder = './faces/'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()


for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    bboxs = detector(img, 1)
    print("Number of faces detected: {}".format(len(bboxs)))

    for k, d in enumerate(bboxs):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))

        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    win.add_overlay(bboxs)
    dlib.hit_enter_to_continue()