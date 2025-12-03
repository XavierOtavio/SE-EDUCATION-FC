import cv2
from picamera2 import Picamera2

cv2.startWindowThread()

picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()

while True:
    im = picam2.capture_array()
    cv2.imshow("cam", im)
    cv2.waitKey(1)