import cv2

def detect_face_on_video_opencvhaar(video_source):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read video stream
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()

        if not ret: break

        frame = cv2.resize(frame, (640, 480))

        # Convert into grayscale
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        # Format of bboxes is: xmin, ymin (top left), xmax, ymax (bottom right)
        for (x, y, w, h) in faces:
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
