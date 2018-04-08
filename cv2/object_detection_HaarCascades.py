import cv2
import numpy as np


#====== object detection with Haar Cascades
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,h,w) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex,ey,eh,ew) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
