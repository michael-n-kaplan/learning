from mtcnn import MTCNN
import cv2

def detect_face_on_video_mtcnn(video_source):
    # Read video stream
    cap = cv2.VideoCapture(video_source)
    detector = MTCNN()
    
    while True:
        ret, frame = cap.read()

        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        
        # Detect face
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(img)
    
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        # Format of bboxes is: xmin, ymin (top left), xmax, ymax (bottom right)
        for face in result:
            xmin = face['box'][0]
            ymin = face['box'][1]
            xmax = xmin + face['box'][2]
            ymax = ymin + face['box'][3]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        cv2.imshow("faces", frame)
        cv2.waitKey(1)
        
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
