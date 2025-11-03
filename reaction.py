#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
GIF_FPS = 15


def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_cv, EMOJI_WINDOW_SIZE)
            frames.append(frame_resized)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def load_static_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} could not be loaded")
    img_resized = cv2.resize(img, EMOJI_WINDOW_SIZE)
    return img_resized


try:
    monkey_finger_mouth_image = load_static_image("photos/monkey_finger_mouth.jpeg")
    monkey_finger_raise_image = load_static_image("photos/monkey_finger_raise.jpg")

    # Load monkey GIF animations
    monkey_mouth_frames = load_gif_frames("photos/hehe.gif")
    monkey_both_hands_frames = load_gif_frames("photos/fk.gif")
    if len(monkey_mouth_frames) == 0:
        raise FileNotFoundError("photos/flight.gif has no frames or could not be loaded")

    if len(monkey_both_hands_frames) == 0:
        raise FileNotFoundError("images/both_hands_raised.gif has no frames or could not be loaded")

    print(f"✅ All monkey images loaded successfully!")
    print(f"   - Monkey finger mouth image loaded")
    print(f"   - Monkey finger raise image loaded")
    print(f"   - Monkey mouth GIF: {len(monkey_mouth_frames)} frames")
    print(f"   - Monkey both hands raised GIF: {len(monkey_both_hands_frames)} frames")

except Exception as e:
    print("❌ Error loading images! Make sure they are in the correct folder and named properly.")
    print(f"Error details: {e}")
    print("\nExpected files in 'images/' folder:")
    print("- images/monkey_finger_mouth.jpeg (finger to mouth)")
    print("- images/monkey_finger_raise.jpg (finger raised)")
    print("- images/flight.gif (tongue out animation)")
    print("- images/both_hands_raised.gif (both hands raised animation)")
    exit()

# Create a blank image for cases where an emoji is missing
blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam capture
print("🎥 Starting webcam capture...")
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if webcam is available
if not cap.isOpened():
    print(
        "❌ Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
    exit()

# Initialize named windows with specific sizes
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Animation Output', cv2.WINDOW_NORMAL)

# Set window sizes and positions
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Animation Output', WINDOW_WIDTH, WINDOW_HEIGHT)

# Position windows side by side
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Animation Output', WINDOW_WIDTH + 150, 100)

print("🚀 Starting monkey gesture detection...")
print("📋 Monkey Gestures:")
print("   - Press 'q' to quit")
print("   🐵 GESTURES:")
print("      • Put finger to mouth = Shh monkey 🤫")
print("      • Raise index finger up = Finger raise monkey ☝️")
print("      • Raise both hands up = Both hands raised monkey 🙌")
print("      • Stick tongue out = Tongue out monkey 👅")
print("   Default: Finger to mouth monkey")

import time

current_animation = "MONKEY_FINGER_MOUTH"  # Default state
animation_frame_index = 0
last_gif_update = time.time()
gif_frame_delay = 1.0 / GIF_FPS  # Delay between GIF frames

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️  Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        small_frame = cv2.resize(frame, (320, 240))

        image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False

        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        results_face = face_mesh.process(image_rgb)

        detected_state = "MONKEY_FINGER_MOUTH"

        if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) == 2:
            both_hands_raised = True

            for hand_landmarks in results_hands.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                hand_raised = (index_tip.y < wrist.y - 0.1 and middle_tip.y < wrist.y - 0.1)

                if not hand_raised:
                    both_hands_raised = False
                    break

            if both_hands_raised:
                detected_state = "MONKEY_BOTH_HANDS_RAISED"
                print("✅ MONKEY: Both hands raised detected!")

        # 2. Check for finger to mouth gesture
        if detected_state == "MONKEY_FINGER_MOUTH" and results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                face_landmarks = results_face.multi_face_landmarks[0]
                mouth_top = face_landmarks.landmark[13]
                mouth_bottom = face_landmarks.landmark[14]
                mouth_center_x = (mouth_top.x + mouth_bottom.x) / 2
                mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2

                distance = ((index_finger_tip.x - mouth_center_x) ** 2 + (
                            index_finger_tip.y - mouth_center_y) ** 2) ** 0.5

                if distance < 0.15:
                    detected_state = "MONKEY_FINGER_MOUTH"
                    print("✅ MONKEY: Finger to mouth detected!")
                    break

        # 3. Check for raised finger gesture (single hand)
        if detected_state == "MONKEY_FINGER_MOUTH" and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get finger landmarks
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Check if index finger is extended and pointing up
                index_extended = index_tip.y < index_mcp.y - 0.1
                index_high = index_tip.y < wrist.y - 0.15
                middle_not_extended = middle_tip.y > index_tip.y + 0.05

                if index_extended and index_high and middle_not_extended:
                    detected_state = "MONKEY_FINGER_RAISE"
                    print("✅ MONKEY: Raised finger detected!")
                    break

        if detected_state == "MONKEY_FINGER_MOUTH" and results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]

            upper_lip_top = face_landmarks.landmark[0]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            lower_lip_bottom = face_landmarks.landmark[17]
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]


            mouth_height = abs(upper_lip.y - lower_lip.y)


            mouth_width = abs(mouth_left.x - mouth_right.x)


            if mouth_width > 0:
                mouth_aspect_ratio = mouth_height / mouth_width
            else:
                mouth_aspect_ratio = 0



            mouth_open = mouth_height > 0.025
            mouth_wide_enough = mouth_aspect_ratio > 0.25

            lip_drop = abs(lower_lip_bottom.y - lower_lip.y)
            tongue_extended = lip_drop > 0.015

            if mouth_open and (mouth_wide_enough or tongue_extended):
                detected_state = "MONKEY_TONGUE_OUT"
                print("✅ MONKEY: Tongue out detected!")
                print(
                    f"   Debug: mouth_height={mouth_height:.4f}, aspect_ratio={mouth_aspect_ratio:.4f}, lip_drop={lip_drop:.4f}")

        current_animation = detected_state
        current_time = time.time()
        if current_time - last_gif_update >= gif_frame_delay:
            animation_frame_index += 1
            last_gif_update = current_time

        if current_animation == "MONKEY_FINGER_MOUTH":
            display_frame = monkey_finger_mouth_image
            state_name = "🐵 Finger to Mouth"
        elif current_animation == "MONKEY_FINGER_RAISE":
            display_frame = monkey_finger_raise_image
            state_name = "🐵 Finger Raised"
        elif current_animation == "MONKEY_BOTH_HANDS_RAISED":
            display_frame = monkey_both_hands_frames[animation_frame_index % len(monkey_both_hands_frames)]
            state_name = "🐵 Both Hands Raised!"
        elif current_animation == "MONKEY_TONGUE_OUT":
            display_frame = monkey_mouth_frames[animation_frame_index % len(monkey_mouth_frames)]
            state_name = "🐵 Tongue Out!"
        else:
            display_frame = monkey_finger_mouth_image
            state_name = "🐵 Shh... Finger to Mouth"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.putText(camera_frame_resized, f'STATE: {state_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Animation Output', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("✅ Application closed successfully!")