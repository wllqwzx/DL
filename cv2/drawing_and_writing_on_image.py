import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
bindings for drawing:
    cv2.line(), 
    cv2.circle() , 
    cv2.rectangle(), 
    cv2.ellipse(), 
    cv2.putText(),
    etc.
'''

img = cv2.imread("test_frcnn_00.jpg", cv2.IMREAD_COLOR)
#======
cv2.line(img, (0,0), (150,150), (255,255,0), 5)

cv2.rectangle(img, (20,30), (200,300), (255,0,255), 5)
cv2.rectangle(img, (100,350), (300,400), (255,0,255), -1)   # -1 means fill

cv2.circle(img, (300,300), 50, (0,0,255), 5)
cv2.circle(img, (600,300), 50, (0,0,255), -1)   # -1 means fill

points = np.array([[500,50],[600,200],[550,400],[400,200]])
points = points.reshape((-1,1,2))
cv2.polylines(img, [points], True, (0,255,255), 3)

cv2.putText(img, "test words", (500,400), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,0,255), 2, cv2.LINE_AA)
#======

cv2.imshow("title", img)
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()
