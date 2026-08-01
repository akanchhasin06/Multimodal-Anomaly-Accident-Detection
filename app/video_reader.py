import cv2

from config import VIDEO_PATH
from config import WINDOW_NAME


'''cv2.VideoCapture() creates an object that opens a video source such as a webcam, 
video file, or IP camera stream,allowing frames to be read sequentially.'''



def run():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Unable to open video")
        return

    while True:
        success, frame=cap.read()
        if not success:
            break;

        #print(frame.shape)
        if  success:
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


