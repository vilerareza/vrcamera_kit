import cv2 as cv
import numpy as np
import dlib


class FaceDetector():
    
    def __init__(self) -> None:

        # dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()


    def detect_face(self, frame):

        try:

            # Conversion to gray
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Face detection
            faces = self.face_detector(img_gray, 0)

            # Do for every face detected
            # for face in faces:

            return None, frame

        finally:
            return faces, frame