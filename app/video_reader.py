import cv2
from ultralytics import YOLO
from config import VIDEO_PATH
from config import WINDOW_NAME


'''cv2.VideoCapture() creates an object that opens a video source such as a webcam, 
video file, or IP camera stream,allowing frames to be read sequentially.'''

model=YOLO("yolov8n.pt")

def run():
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps=cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    if fps == 0:
        fps = 30
    delay=int(1000/fps)

    if not cap.isOpened():
        print("Unable to open video")
        return

    frame_number=0

    while True:
        success, frame=cap.read()
        if not success:
            break;

        results = model(frame, verbose=False)
        print(dir(results[0]))

        if  success:
            text = f"FPS : {fps:.2f}"
            frame_text = f"Frame : {frame_number}"

            cv2.putText(
                frame,    #Image to write
                text ,     #Text tp write
                (10,30),    #Position(x,y)
                cv2.FONT_HERSHEY_SIMPLEX,  #Font
                1,          #Font size
                (0,0,0),    #Color
                2           #Thickness
            )

            cv2.putText(
                frame, 
                frame_text, 
                (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0,0,0),
                2
            )

            frame_number+=1

            cv2.rectangle(
                frame,
                (100,100),
                (300,250),
                (0,255,0), #BGR format
                2           #Thickness of the border
            )

            frame_text2="Car"

            cv2.putText(
                frame , 
                frame_text2 , 
                (100,90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0,255,0),
                2
            )

            cv2.imshow("Frame", frame)
            key=cv2.waitKey(delay)    #After showing the current frame, wait 30 milliseconds before moving to the next frame

            if key==ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()



