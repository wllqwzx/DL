import dlib
import os
import glob
from skimage import io


detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

faces_folder = './faces/'

for f in glob.glob(os.path.join(faces_folder, "*.jp*g")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    bboxs = detector(img, 1)
    print("Number of faces detected: {}".format(len(bboxs)))

    win.add_overlay(bboxs)
    dlib.hit_enter_to_continue()
