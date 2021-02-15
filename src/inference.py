import cv2 
import dlib
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils

class Face_Detection():

    def __init__(self):
        self.predictor = 'models/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
    

    def smile(self, mouth):
        A = dist.euclidean(mouth[3], mouth[9])
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[4], mouth[8])
        avg = (A+B+C)/3
        D = dist.euclidean(mouth[0], mouth[6])
        mar=avg/D
        return mar

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


    def video_analyze(self):
        EYE_AR_THRESH = 0.25

        predictor = dlib.shape_predictor(self.predictor)

        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        vs = cv2.VideoCapture('videos/test2.mp4')

        frame_count = 0

        while True:
            try:
                
                ret, frame = vs.read()
                if not ret:
                    break
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                frame_count += 1
                print(frame_count)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    mouth= shape[mStart:mEnd]
                    mar= self.smile(mouth)
                    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                
                    if mar <= .3 or mar > .38 :
                        print('similing')
                    else:
                        print('not smiling')
                    
                    if ear < EYE_AR_THRESH:
                        print('eye is closed')

                    else:
                        print('eye is open')
                cv2.imshow("face detector" , frame)

            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":

    face_detection = Face_Detection()
    face_detection.video_analyze()
        

        

