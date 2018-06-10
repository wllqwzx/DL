import dlib
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print("image width:", cap.get(3))
print("image height:", cap.get(4))
#cap.set(3, 640) # set video height to 640
#cap.set(4, 360) # set video width to 360

predictor_path = './bin/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# 3d reference points:
ref_pts = np.array([
    [6.825897, 6.760612, 4.402142],     # 
    [1.330353, 7.122144, 6.903745],     #
    [-1.330353, 7.122144, 6.903745],    #
    [-6.825897, 6.760612, 4.402142],    #
    [5.311432, 5.485328, 3.987654],     #
    [1.789930, 5.393625, 4.413414],     #
    [-1.789930, 5.393625, 4.413414],    #
    [-5.311432, 5.485328, 3.987654],    #
    [2.005628, 1.409845, 6.165652],     #
    [-2.005628, 1.409845, 6.165652],    #
    [2.774015, -2.080775, 5.048531],    #
    [-2.774015, -2.080775, 5.048531],   #
    [0.000000, -3.116408, 6.097667],    #
    [0.000000, -7.415691, 4.070434]     #
], dtype=np.float32)


ref_3d_bbox = np.array([
    [10,10,10],
    [10,10,-10],
    [10,-10,-10],
    [10,-10,10],
    [-10,10,10],
    [-10,10,-10],
    [-10,-10,-10],
    [-10,-10,10],

    [0, 1.409845, 6.165652],    #
    [0, 1.409845, 30]           # 
], dtype=np.float32)


while cap.isOpened():
    ret, frame = cap.read()

    faces = detector(frame,1)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(frame, face)

        for i in range(landmarks.num_parts):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0,0,255), -1)

        # 2d points on image
        image_pts = np.array([
            [landmarks.part(17).x, landmarks.part(17).y],
            [landmarks.part(21).x, landmarks.part(21).y],
            [landmarks.part(22).x, landmarks.part(22).y],
            [landmarks.part(26).x, landmarks.part(26).y],
            [landmarks.part(36).x, landmarks.part(36).y],
            [landmarks.part(39).x, landmarks.part(39).y],
            [landmarks.part(42).x, landmarks.part(42).y],
            [landmarks.part(45).x, landmarks.part(45).y],
            [landmarks.part(31).x, landmarks.part(31).y],
            [landmarks.part(35).x, landmarks.part(35).y],
            [landmarks.part(48).x, landmarks.part(48).y],
            [landmarks.part(54).x, landmarks.part(54).y],
            [landmarks.part(57).x, landmarks.part(57).y],
            [landmarks.part(8).x, landmarks.part(8).y]
        ], dtype=np.float32)

        # apporximate intrinsic params
        focal_length = frame.shape[1]
        center = (frame.shape[1]/2, frame.shape[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0,0,1]
        ])

        # assume no distortion
        dist_coefs = np.zeros((4,1))

        success, r, t = cv2.solvePnP(ref_pts, image_pts, camera_matrix, dist_coefs, flags=cv2.SOLVEPNP_ITERATIVE)
        projected_3d_bbox, _ = cv2.projectPoints(ref_3d_bbox, r, t, camera_matrix, dist_coefs)
        projected_3d_bbox = projected_3d_bbox.reshape(10,2)
        
        # draw 3d bbox
        cv2.line(frame, tuple(projected_3d_bbox[0]), tuple(projected_3d_bbox[1]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[1]), tuple(projected_3d_bbox[2]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[2]), tuple(projected_3d_bbox[3]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[3]), tuple(projected_3d_bbox[0]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[4]), tuple(projected_3d_bbox[5]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[5]), tuple(projected_3d_bbox[6]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[6]), tuple(projected_3d_bbox[7]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[7]), tuple(projected_3d_bbox[4]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[0]), tuple(projected_3d_bbox[4]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[1]), tuple(projected_3d_bbox[5]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[2]), tuple(projected_3d_bbox[6]), (0,0,255))

        # draw a line from nose center
        cv2.line(frame, tuple(projected_3d_bbox[3]), tuple(projected_3d_bbox[7]), (0,0,255))
        cv2.line(frame, tuple(projected_3d_bbox[8]), tuple(projected_3d_bbox[9]), (0,0,255))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

