import cv2
import numpy as np
import matplotlib.pyplot as plt


#====== feature matching
img1 = cv2.imread("test_frcnn_00.jpg")[100:400,400:700]
img2 = cv2.imread("test_frcnn_00.jpg")[200:500,450:750]

orb = cv2.ORB_create()
kp1, descriptor1 = orb.detectAndCompute(img1, None)
kp2, descriptor2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptor1, descriptor2)
matches = sorted(matches, key=lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)

while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()

