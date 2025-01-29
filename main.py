import cv2 as cv
import numpy as np

# Load pre-trained network
MODEL_PATH = "graph_opt.pb"
net = cv.dnn.readNetFromTensorflow(MODEL_PATH)

# Model parameters
INPUT_WIDTH = 368
INPUT_HEIGHT = 368
CONFIDENCE_THRESHOLD = 0.2

# Body parts and pose pairs for the model
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

def pose_estimation(frame):
    """
    Perform pose estimation on a single frame.

    Args:
        frame (np.array): Input image.

    Returns:
        np.array: Frame with pose landmarks and connections drawn.
    """
    frame_height, frame_width = frame.shape[:2]

    # Preprocess the image
    input_blob = cv.dnn.blobFromImage(
        frame, 1.0, (INPUT_WIDTH, INPUT_HEIGHT), (127.5, 127.5, 127.5), swapRB=True, crop=False
    )
    net.setInput(input_blob)
    output = net.forward()
    output = output[:, :19, :, :]  # Only consider the first 19 parts

    points = []
    for i in range(len(BODY_PARTS)):
        heatmap = output[0, i, :, :]
        _, confidence, _, point = cv.minMaxLoc(heatmap)

        if confidence > CONFIDENCE_THRESHOLD:
            x = int((frame_width * point[0]) / output.shape[3])
            y = int((frame_height * point[1]) / output.shape[2])
            points.append((x, y))
        else:
            points.append(None)

    # Draw pose landmarks and connections
    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]

        if points[id_from] and points[id_to]:
            cv.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv.ellipse(frame, points[id_from], (5, 5), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[id_to], (5, 5), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return frame

def main():
    """
    Main function to capture video from webcam and apply pose estimation.
    """
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        estimated_frame = pose_estimation(frame)
        cv.imshow("Pose Estimation", estimated_frame)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
