import cv2
import numpy as np

config = "yolov3.cfg"  # YOLO configuration file
weights = "yolov3.weights"  # YOLO pre-trained weights
classes_file = "yolov3.txt"  # File containing class names
target_classes = [46, 47, 49]  # Classes to detect

# Reassign classes and COLORS as global variables
global classes, COLORS

# Read the class names from the classes file
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    # Get the names of the output layers of the network
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # Draw bounding box and label on the image
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def analyze(image):
    # Analyze the input image for object detection

    # Get the width and height of the image
    Width = image.shape[1]
    Height = image.shape[0]

    # Set the scale for the image
    scale = 0.00392

    # Load the pre-trained YOLO model
    net = cv2.dnn.readNet(weights, config)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(get_output_layers(net))

    # Initialize lists to store the class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Set the confidence and non-maximum suppression thresholds
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Iterate over each output layer
    for out in outs:
        # Iterate over each detection
        for detection in out:
            # Get the scores and class ID
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the confidence is above the threshold and the class is in the target classes
            if confidence > conf_threshold: #and class_id in target_classes:
                # Calculate the bounding box coordinates
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                # Add the class ID, confidence, and bounding box to the respective lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Print the detected class and confidence
                print(f"Detected class: {classes[class_id]}, Confidence: {confidence}")

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Iterate over the selected indices
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        # Get the coordinates of the bounding box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # Draw the bounding box and label on the image
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # Return the annotated image
    return image

# Initialize a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

iteration = 0;

while True:
    print(f"Iteration: {iteration}")

    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    #cv2.imshow('Input', frame)
    result = analyze(frame)
    cv2.imshow("object detection", result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    iteration = iteration + 1

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()
