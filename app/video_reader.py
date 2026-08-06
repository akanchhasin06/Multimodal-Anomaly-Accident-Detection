import cv2
from ultralytics import YOLO
from config import VIDEO_PATH
from config import WINDOW_NAME

BOX_COLOR = (0,255,0)
TEXT_COLOR = (0,255,0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


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

        frame_number+=1

        if frame_number % 3 == 0:
            results = model(frame, verbose=False)
            boxes = results[0].boxes


        results = model(frame, verbose=False)
        boxes = results[0].boxes

        text = f"FPS : {fps:.2f}"
        frame_text = f"Frame : {frame_number}"
        
        cv2.putText(
        frame,    #Image to write
        text ,     #Text tp write
        (10,30),    #Position(x,y)
        FONT,  #Font
        1,          #Font size
        (0,0,0),    #Color
        2           #Thickness
        )
        
        cv2.putText(
        frame, 
        frame_text, 
        (10,70),
        FONT, 
        1,
        (0,0,0),
        2
        )
        
        
        

        for box in boxes:
            print(box)
            print("-------------------")
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                BOX_THICKNESS
            )

            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            confidence = float(box.conf[0])
            label = f"{class_name} {confidence*100:.1f}%"


                    
            cv2.putText(
            frame , 
            label , 
            (x1,y1-10), 
            FONT, 
            1,
            (0,255,0),
            2
            )


        if len(boxes) > 0:
            print("Objects detected:", len(boxes))

            cv2.imshow("Frame", frame)
            key=cv2.waitKey(delay)    #After showing the current frame, wait 30 milliseconds before moving to the next frame

            if key==ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()



