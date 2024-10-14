import time
import os
import urllib.request
import cv2
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib.pyplot as plt
import json

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from json import dumps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.makedirs('models', exist_ok=True)
face_model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
face_model_path = os.path.join('models', 'face_landmarker.task')
urllib.request.urlretrieve(face_model_url, face_model_path)

hand_model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
hand_model_path = os.path.join('models', 'hand_landmarker.task')
urllib.request.urlretrieve(hand_model_url, hand_model_path)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, face_detection_result, hand_detection_result):
    annotated_image = np.copy(rgb_image)

    # Draw face landmarks
    if face_detection_result and face_detection_result.face_landmarks:
        for face_landmarks in face_detection_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    # Draw hand landmarks
    if hand_detection_result and hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks_proto,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image


def draw_solid_sphere(image, center, radius, color=(255, 0, 0)):
    # Draw a solid sphere on the image
    num_points = 100
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.outer(np.sin(phi), np.cos(theta)) + center[0]
    y = radius * np.outer(np.sin(phi), np.sin(theta)) + center[1]
    z = radius * np.outer(np.cos(phi), np.ones_like(theta)
                          ) + 0  # Assuming a fixed z-plane

    # Project 3D points to 2D
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            pt1 = (int(x[i, j]), int(y[i, j]))
            pt2 = (int(x[i + 1, j]), int(y[i + 1, j]))
            pt3 = (int(x[i, j + 1]), int(y[i, j + 1]))
            pt4 = (int(x[i + 1, j + 1]), int(y[i + 1, j + 1]))

            # Draw triangles to create the sphere effect
            cv2.fillConvexPoly(image, np.array([pt1, pt2, pt4]), color)
            cv2.fillConvexPoly(image, np.array([pt1, pt3, pt4]), color)


# Create face landmarker
face_base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# Create hand landmarker
hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options,
                                            num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Initialize webcam
cap = cv2.VideoCapture(0)

face_detection_result = None
hand_detection_result = None

# Initialize sphere properties
sphere_radius = 50
sphere_position = (100, 100)  # Initial position of the sphere

# New variables for sticky feature
is_sphere_stuck = False
stuck_position = None
stuck_time = 0
position_history = []
STICK_THRESHOLD = 30  # pixels
STICK_TIME = 1.0  # seconds
UNSTICK_TIMEOUT = 20.0  # seconds


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, retrying...")
        continue  # Retry capturing the frame instead of breaking the loop

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect face landmarks
    face_detection_result = face_detector.detect(mp_image)

    # Detect hand landmarks
    hand_detection_result = hand_detector.detect(mp_image)

    # Serialize face and hand landmarks and save to disk
    frame_data = {
        "face_landmarks": [],
        "hand_landmarks": []
    }
    jaw_open_values = []  # List to keep track of jaw open values

    if face_detection_result.face_landmarks:
        for blendshapes in face_detection_result.face_blendshapes:
            for blendshape in blendshapes:
                if blendshape.category_name == 'jawOpen':
                    # Keep the jaw open value
                    jaw_open_values.append(blendshape.score)

    if face_detection_result.face_landmarks:
        for face_landmarks in face_detection_result.face_landmarks:
            frame_data["face_landmarks"] = [
                {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "presence": landmark.presence
                } for landmark in face_landmarks
            ]

    if hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            frame_data["hand_landmarks"].append([
                {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                } for landmark in hand_landmarks
            ])

    # Save frame data to a JSON file
    with open('frame_data.json', 'w') as f:
        json.dump(frame_data, f)

    # Check the last 40 jaw open values
    if len(jaw_open_values) > 40:
        jaw_open_values = jaw_open_values[-40:]  # Keep only the last 40 values

    # Adjust sphere radius based on jaw open values
    if jaw_open_values and jaw_open_values[-1] > 0.2:
        sphere_radius = 100  # Zoom in when jaw is open
    else:
        sphere_radius = 50  # Default radius

    # Update sphere position based on hand landmark
    if hand_detection_result.hand_landmarks and not is_sphere_stuck:
        # Use the tip of the index finger (landmark 8) to control the sphere
        index_finger_tip = hand_detection_result.hand_landmarks[0][8]
        new_position = (
            int(index_finger_tip.x * frame.shape[1]),
            int(index_finger_tip.y * frame.shape[0])
        )

        # Add new position to history
        position_history.append(new_position)
        if len(position_history) > 10:  # Keep only last 10 positions
            position_history.pop(0)

        # Check if the hand has been relatively still
        if all(np.linalg.norm(np.array(new_position) - np.array(pos)) < STICK_THRESHOLD for pos in position_history):
            if stuck_time == 0:
                stuck_time = time.time()
            elif time.time() - stuck_time > STICK_TIME:
                is_sphere_stuck = True
                stuck_position = new_position
                print("Sphere stuck!")
        else:
            stuck_time = 0

        sphere_position = new_position
    elif not hand_detection_result.hand_landmarks and is_sphere_stuck:
        # If hand is not detected and sphere is stuck, keep it at the stuck position
        sphere_position = stuck_position
    elif hand_detection_result.hand_landmarks and is_sphere_stuck:
        # If hand is detected and sphere is stuck, check if it's far from the stuck position
        index_finger_tip = hand_detection_result.hand_landmarks[0][8]
        current_position = (
            int(index_finger_tip.x * frame.shape[1]),
            int(index_finger_tip.y * frame.shape[0])
        )
        if time.time() - stuck_time >= 5 and np.linalg.norm(np.array(current_position) - np.array(stuck_position)) > STICK_THRESHOLD * 2:
            is_sphere_stuck = False
            stuck_position = None
            stuck_time = 0
            position_history.clear()
            print("Sphere unstuck!")

    # Create a dark view instead of showing the face
    # Create a dark image with the same shape as the frame
    dark_view = np.zeros_like(frame)

    # Draw landmarks on the dark image
    annotated_image = draw_landmarks_on_image(
        dark_view, face_detection_result, hand_detection_result)

    # Draw the sphere at the updated position
    draw_solid_sphere(annotated_image, sphere_position, sphere_radius, color=(
        0, 255, 0) if is_sphere_stuck else (255, 0, 0))

    # Display the resulting frame
    cv2.imshow('Face and Hand Landmarks with Controlled Sphere',
               cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
