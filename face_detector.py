import numpy as np
import dlib
import cv2 as cv

class FaceDetector():
    
    def __init__(self) -> None:

        # dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()


    def detect_face(self, frame):

        #try:

        np_frame = np.asarray(bytearray(frame))

        if np_frame.any():
            #try:
            img = cv.imdecode(np_frame, 1)
            #except:
            #    pass

        # Conversion to gray
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Face detection
        faces = self.face_detector(img_gray, 0)

            # Do for every face detected
            # for face in faces:

        return None, frame

        # finally:
        #     return faces, frame