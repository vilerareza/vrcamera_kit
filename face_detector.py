import os
import numpy as np
import dlib
import cv2 as cv
import time
from datetime import datetime
import pickle
import sqlite3
import base64


class FaceDetector():
    
    def __init__(self, db_path, db_table, img_dir) -> None:

        # dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()
        self.db_path = db_path
        self.db_table = db_table
        self.img_dir = img_dir


    def get_face_rect(self, img, dlib_rect):
        x1 = max(0, dlib_rect.left())
        y1 = max(0, dlib_rect.top())
        x2 = min(dlib_rect.right(), img.shape[1])
        y2 = min(dlib_rect.bottom(), img.shape[0])
        return [(x1,y1), (x2,y2)]


    def insert_to_db(self, 
                     db_path, 
                     db_table, 
                     det_type, 
                     det_datetime, 
                     img_file):
        try:
            cmd = f"""insert into {db_table}(det_type, det_datetime, img_file) values (?, ?, ?)"""
            with sqlite3.connect(db_path) as conn:
                conn.execute(cmd, (det_type, det_datetime, img_file))
                conn.commit()
            print ('inserted to db')
        except Exception as e:
            print (f'[ERROR] {e}: insert_to_db failed')


    def detect_face(self, frame_raw, bbox = True):
        
        # Forming the image array
        np_frame = np.asarray(bytearray(frame_raw))
        img = cv.imdecode(np_frame, 1)

        # Conversion to gray
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Face detection
        faces = self.face_detector(img_gray, 0)

        # time stamp
        date_time = datetime.now().strftime('%d-%m-%y, %H:%M')
        
        print (len(faces), faces)
        
        # Process on face exists
        if len(faces) > 0:
        
            for face in faces:
            
                # Draw rectangle on faces
                if bbox:
                    start_pt, end_pt = self.get_face_rect(img, face)
                    cv.rectangle(img, 
                                start_pt,
                                end_pt, 
                                (0,255,0), 
                                3)

                # Saving the frame image
                filename = str(time.time()).replace('.','')
                cv.imwrite(os.path.join(self.img_dir, f'{filename}.png'), img)

                # Insert the face detection to database
                self.insert_to_db(db_path = self.db_path, 
                                  db_table = self.db_table, 
                                  det_type='face',
                                  det_datetime = date_time,
                                  filename = filename
                                  )
                
                # cv.imwrite(f'face_{str(time.time())[-5:]}.png', img)
        
        return faces, img

        # finally:
        #     return faces, frame