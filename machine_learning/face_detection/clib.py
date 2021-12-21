import dlib
from imutils import face_utils
import cv2


def detect_face_on_video_dlib(video_source):
    
    # Read video stream
    cap = cv2.VideoCapture(video_source)

    detector = dlib.get_frontal_face_detector()
    
    while True:
        ret, frame = cap.read()

        if not ret: break

        frame = cv2.resize(frame, (640, 480))

        # Convert into grayscale
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(img, 1)
        
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        # Format of bboxes is: xmin, ymin (top left), xmax, ymax (bottom right)
        for i in faces:
            (x, y, w, h) = face_utils.rect_to_bb(i) 
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        cv2.imshow("faces", frame)
        cv2.waitKey(1)
        
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
