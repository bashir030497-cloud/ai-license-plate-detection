import cv2
# Import OpenCV library
# Used for webcam access, drawing boxes, and showing video window

from ultralytics import YOLO
# Import YOLO class from Ultralytics
# Used to load and run YOLOv8 model


# Load the trained YOLOv8 model (license plate detection model)
# best.pt is the trained weights file
model = YOLO(r"C:\Users\BASHIR AHMAD\Desktop\AI project\best.pt")


# Open the webcam
# 0 means default camera
video = cv2.VideoCapture(0)


# Start an infinite loop to read webcam frames continuously
while True:

    # Read a frame from the webcam
    ret, frame = video.read()
    # ret = True if frame is successfully captured
    # frame = the actual image from webcam

    # If frame is not captured properly, stop the loop
    if not ret:
        break


    # Run YOLOv8 detection on the current frame
    # [0] gives the first result (single frame result)
    results = model(frame)[0]


    # Loop through all detected bounding boxes
    for box in results.boxes:

        # Get the detected class ID
        # Example: 0 = license plate
        class_id = int(box.cls[0])

        # Only process license plate class (class 0)
        # Skip all other detected objects
        if class_id != 0:
            continue


        # Get bounding box coordinates
        # (x1, y1) = top-left corner
        # (x2, y2) = bottom-right corner
        x1, y1, x2, y2 = map(int, box.xyxy[0])


        # Get confidence score of the detection
        confidence = float(box.conf[0])


        # Draw a green rectangle around the license plate
        # (0, 255, 0) = green color
        # 2 = thickness of rectangle line
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Display label and confidence score above the bounding box
        cv2.putText(
            frame,
            f"Plate {confidence:.2f}",     # Text to display
            (x1, y1 - 10),                 # Position of text
            cv2.FONT_HERSHEY_SIMPLEX,      # Font type
            0.7,                           # Font size
            (0, 255, 0),                   # Text color (green)
            2                              # Text thickness
        )


    # Show the output window with detected license plates
    cv2.imshow("License Plate Detection", frame)


    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam after exiting the loop
video.release()

# Close all OpenCV windowsqq
cv2.destroyAllWindows()
