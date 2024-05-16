import cv2
import numpy as np
import socket
import time
import subprocess


config = "yolov3.cfg"
weights = "yolov3.weights"
classes_file = "yolov3_tradotto.txt"

target_classes1 = ["Mela"]
target_classes2 = ["Banana"]
target_classes3 = ["Arancia"]

target_classes = {
    tuple(target_classes1): 48,
    tuple(target_classes2): 47,
    tuple(target_classes3): 50
    # Aggiungi altre classi se necessario
}

# The remote host (ROBOT)...................
HOST = "192.168.10.2"  
# UR control port...........................
PORT = 30002  

global classes, COLORS

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to send a UR script command to the robot.............................................
def URsetOuts (HOST, PORT, DO0, DO1, DO2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    txt = "sec setOuts():\n  set_digital_out(0," + str(DO0) + ")\n  set_digital_out(1," + str(DO1) + ")\n  set_digital_out(2," + str(DO2) + ")\nend\nsetOuts()"
    msg = txt.encode("utf8")
    print("Sending:", msg)
    s.send(msg)
    data = s.recv(1024)
    s.close()
    print("Received", repr(data))
#.......................................
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers
#.......................................
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#..........................................
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

iteration = 0
#..........................................   2

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

    # Flag to check if any target object is detected
    target_detected = False

    # Iterate over each output layer
    for out in outs:
        # Iterate over each detection
        for detection in out:
            # Get the scores and class ID
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
          

            # Check if the confidence is above the threshold and the class is in the target classes
            if confidence > conf_threshold: 
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

                # Set the flag to true as target object is detected
                target_detected = True

                # Print the detected class and confidence
                print(f"Detected class: {classes[class_id] }, Confidence: {confidence}")

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
        #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # If a target object is detected, write "Nome oggetto + rilevato"
    if target_detected:
        cv2.putText(image, f"{classes[class_ids[0]]} Individuato", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

 
    if target_detected:
        detected_class = classes[class_ids[0]]

        if detected_class in target_classes1:
            print("Apple")
            URsetOuts(HOST, PORT, True, False, False)
            time.sleep(2)
            URsetOuts(HOST, PORT, False, False, False)
            
            

        elif detected_class in target_classes2:
            print("Banana")
            URsetOuts(HOST, PORT, False, True, False)
            time.sleep(2)
            URsetOuts(HOST, PORT, False, False, False)
            
            
        elif detected_class in target_classes3:
            print("Orange")
            URsetOuts(HOST, PORT, False, False, True)
            time.sleep(2)
            URsetOuts(HOST, PORT, False, False, False)
            
            

            
        else:
            print("Unknown class")
    else:
        print("Nessuna classe rilevata con confidenza sufficiente")
   

        
    # Return the annotated image
    return image

#..........................................



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

    iteration = iteration +1
       

cap.release()
cv2.destroyAllWindows()