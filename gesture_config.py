import cv2
import mediapipe as mp
import math
import pyautogui
import time

current_pose = ""
previous_pose = ""
zone = ""

previous_positions = []

index_image_pos = 0, 0
smoothing_factor = 2
framerate = 30

delay = 1 / framerate
screen_width, screen_height = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get the positions of the hand keypoints
            palm_pos = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y

            thumb_pos = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
            index_pos = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            middle_pos = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
            ring_pos = hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y
            pinky_pos = hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y

            thumb_joint = hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y
            index_joint = hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y
            middle_joint = hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y
            ring_joint = hand_landmarks.landmark[15].x, hand_landmarks.landmark[15].y
            pinky_joint = hand_landmarks.landmark[19].x, hand_landmarks.landmark[19].y

            thumb_base = hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y
            index_base = hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y
            middle_base = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y
            ring_base = hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y
            pinky_base = hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y

            # Calculate the distances between the fingertips and palm
            thumb_dist = math.sqrt(pow(palm_pos[0] - thumb_pos[0], 2) + pow(palm_pos[1] - thumb_pos[1], 2))
            index_dist = math.sqrt(pow(palm_pos[0] - index_pos[0], 2) + pow(palm_pos[1] - index_pos[1], 2))
            middle_dist = math.sqrt(pow(palm_pos[0] - middle_pos[0], 2) + pow(palm_pos[1] - middle_pos[1], 2))
            ring_dist = math.sqrt(pow(palm_pos[0] - ring_pos[0], 2) + pow(palm_pos[1] - ring_pos[1], 2))
            pinky_dist = math.sqrt(pow(palm_pos[0] - pinky_pos[0], 2) + pow(palm_pos[1] - pinky_pos[1], 2))

            # Calculate distance between index base and thumb tip
            index_thumb_base_dist = math.sqrt(pow(index_base[0] - thumb_pos[0], 2) + pow(index_base[1] - thumb_pos[1], 2))

            previous_pose = current_pose

            # Identify hand poses
            if index_joint[1] > index_pos[1] and index_dist > thumb_dist and middle_pos[1] > middle_base[1] and middle_dist < 0.3:
                hand_pose = "index finger"
            elif middle_pos[1] < middle_base[1] and index_joint[1] > index_pos[1] and ring_joint[1] < ring_pos[1] and pinky_joint[1] < pinky_pos[1] and ring_dist < 0.3:
                hand_pose = "two fingers v"
            elif ring_pos[1] < ring_base[1] and index_joint[1] > index_pos[1] and ring_joint[1] > ring_pos[1] and pinky_joint[1] < pinky_pos[1] and pinky_dist < 0.3:
                hand_pose = "three fingers v"
            elif pinky_pos[1] < pinky_base[1] and index_joint[1] > index_pos[1] and ring_joint[1] > ring_pos[1] and pinky_joint[1] > pinky_pos[1] and pinky_dist > 0.3:
                hand_pose = "four fingers v"
            else:
                hand_pose = "unknown"

            current_pose = hand_pose

            # Set value on first occurrence of pose
            if current_pose != previous_pose and hand_pose != "unknown":
                index_image_pos = index_pos

            # Identify zones
            if index_image_pos[0] < 0.75 * index_pos[0]:
                zone = "left"
            elif index_image_pos[0] > 1.25 * index_pos[0]:
                zone = "right"
            elif index_image_pos[1] > 1.3 * index_pos[1]:
                zone = "top"
            elif index_image_pos[1] < 0.8 * index_pos[1]:
                zone = "bottom"
            else:
                zone = "unknown"

            # Actions
            if hand_pose == "index finger":
                subarea_width = 640
                subarea_height = 480

                normalized_x = index_pos[0] * image_width - subarea_width / 2
                normalized_y = index_pos[1] * image_height - subarea_height / 2

                screen_x = screen_width - normalized_x * screen_width / subarea_width
                screen_y = normalized_y * screen_height / subarea_height

                previous_positions.append((screen_x, screen_y))

                if len(previous_positions) > smoothing_factor:
                    previous_positions.pop(0)

                average_x = sum([pos[0] for pos in previous_positions]) / len(previous_positions)
                average_y = sum([pos[1] for pos in previous_positions]) / len(previous_positions)

                pyautogui.moveTo(average_x, average_y)

                if index_thumb_base_dist < 0.1:
                    pyautogui.click()

            elif hand_pose == "two fingers v":
                if zone == "top":
                    pyautogui.scroll(15)
                elif zone == "bottom":
                    pyautogui.scroll(-15)
                elif zone == "left":
                    pyautogui.hscroll(5)
                elif zone == "right":
                    pyautogui.hscroll(-5)

            elif hand_pose == "three fingers v":
                if zone == "left":
                    pyautogui.hotkey('ctrl', 'option', 'shift', 'left')
                    time.sleep(1)
                elif zone == "right":
                    pyautogui.hotkey('ctrl', 'option', 'shift', 'right')
                    time.sleep(1)

            elif hand_pose == "four fingers v":
                if zone == "left":
                    pyautogui.hotkey('ctrl', 'left')
                    time.sleep(1)
                elif zone == "right":
                    pyautogui.hotkey('ctrl', 'right')
                    time.sleep(1)

    cv2.imshow('Gesture Window', cv2.flip(image, 1))

    time.sleep(delay)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
