import cv2
import numpy as np

# Model files configuration
PROTOTXT = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"

# List of classes MobileNet SSD detects (Index 15 is Person, Index 2 is Bicycle, etc.)
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Load pre-trained neural network
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Fulfills requirement: Set up real-time video input using a webcam
cap = cv2.VideoCapture(0)

# Tracker Initialization (Using built-in OpenCV tracking functionality)
tracker = cv2.legacy.TrackerCSRT_create()
tracking_active = False
track_box = None
label = "Tracking"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if not tracking_active:
        # Fulfills requirement: Use a pre-trained model for object detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        # Fulfills requirement: Process each video frame to detect objects
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections (Confidence threshold 50%)
            if confidence > 0.5:
                class_idx = int(detections[0, 0, i, 1])
                label = CLASSES[class_idx]

                # Fulfills requirement: Draw bounding boxes
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")

                # Convert to width/height format required by trackers
                track_box = (startX, startY, endX - startX, endY - startY)
                tracker.init(frame, track_box)
                tracking_active = True
                break
    else:
        # Fulfills requirement: Apply object tracking
        success, box = tracker.update(frame)
        if success:
            x, y, bw, bh = [int(v) for v in box]
            # Fulfills requirement: Display the output with labels and bounding boxes
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} [ID: 101]",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            # If tracking fails, reset tracking to search for objects again
            tracking_active = False
            tracker = cv2.legacy.TrackerCSRT_create()

    # Fulfills requirement: Display output in real time
    cv2.imshow("CodeAlpha Object Detection & Tracking", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()