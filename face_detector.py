import numpy as np
import dlib
import cv2 as cv

class FaceDetector():
    
    def __init__(self) -> None:

        # dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()


    def detect_face(self, frame_raw, bbox = True):
        
        # Forming the image array
        np_frame = np.asarray(bytearray(frame_raw))
        img = cv.imdecode(np_frame, 1)

        # Conversion to gray
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Face detection
        faces = self.face_detector(img_gray, 0)

        print (len(faces), faces)
        
        # Do for every face detected
        # Draw face
        for face in faces:
            cv.rectangle(img, 
                         face, 
                         (0,255,0), 
                         3)

        return faces, img

        # finally:
        #     return faces, frame