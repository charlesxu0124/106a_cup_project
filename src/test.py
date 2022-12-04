from pipeline import pixel_to_robot
from image.video_capture import VideoCapture
import cv2
import numpy as np
K = np.array([[604.026546, 0.000000, 331.477939],
[0.000000, 602.778325, 214.438981],
[0.000000, 0.000000, 1.000000]])

dist = np.array([0.171886, -0.299336, -0.008033, 0.01494, 0.000000])


# if __name__ == 'main':
cap = VideoCapture('/dev/video0')
img = cap.read()
cv2.imshow('distorted image', img)
undistor_img = cv2.undistort(img, cameraMatrix=K, distCoeffs=dist, newCameraMatrix=K)
cv2.imshow('undistorted image', undistor_img)
cv2.waitKey(0)
pixels = input('input pixel: ').split(", ")
pixels = np.array([int(pixels[0]), int(pixels[1])])
#pixel = np.array([278,267])
x,y=pixel_to_robot(pixels)
print(x, ', ', y)
