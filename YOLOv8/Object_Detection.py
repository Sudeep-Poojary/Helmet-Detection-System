import cv2
from ultralytics import YOLO

# Replace 'path/to/your/model.pt' with your model path
model = YOLO("D:\Object_Detection_YOLOv8\yolov8n.pt")

# Define source (0 for webcam, path to video file for video)
source = 0

# Define confidence threshold (optional)
conf_thres = 0.5

# Define Non-Maximum Suppression (NMS) threshold (optional)
nms_thres = 0.5

cap = cv2.VideoCapture(source)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # If frame is read correctly
  if ret:
    # Get predictions
    results = model(frame)

    # Loop through detections
    for result in results.pandas().xyxy[0]:
      # Filter by confidence threshold (optional)
      if result['conf'] > conf_thres:
        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = int(result['xmin']), int(result['ymin']), int(result['xmax']), int(result['ymax'])
        class_id = int(result['name'])
        class_name = model.names[class_id]

        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({result['conf']:.2f})", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop if frame is not read correctly
  else:
    break

# Release capture and close window
cap.release()
cv2.destroyAllWindows()