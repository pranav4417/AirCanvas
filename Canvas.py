import cv2
import numpy as np
import mediapipe as mp
from playsound import playsound
import threading
import collections

# Play audio asynchronously
def play_ding():
    threading.Thread(target=playsound, args=('/Users/pranavkandakurthi/Downloads/Labsheets/ding.wav',), daemon=True).start()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,  # Increased for better accuracy
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Smoothing for drawing
point_history = collections.deque(maxlen=5)
prev_x, prev_y = 0, 0
draw_color = (0, 0, 255)  # Default red
color_selected = False
color_selection_frames = 0
COLOR_SELECTION_THRESHOLD = 15  # Frames to confirm color selection

def draw_color_boxes(img):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, color in enumerate(colors):
        x1, y1, x2, y2 = i * 80, 0, (i + 1) * 80, 60
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return colors

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for tip in tips:
        if tip == 4:  # Thumb
            if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 2].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
    return fingers

def is_fist(hand_landmarks):
    return sum(fingers_up(hand_landmarks)) == 0

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    colors = draw_color_boxes(frame)
    color_changed_this_frame = False
    candidate_color = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark
            h, w, _ = frame.shape

            # Thumb tip (landmark 4)
            tx = int(lm[4].x * w)
            ty = int(lm[4].y * h)
            # Index tip for drawing (landmark 8)
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            fingers = fingers_up(handLms)

            # Visual feedback for thumb
            cv2.circle(frame, (tx, ty), 10, (255, 255, 255), 2)
            
            # Color selection with thumb
            if ty < 60:
                box_index = tx // 80
                if box_index < len(colors):
                    # Thumb selection confirmation: thumb must stay in box
                    if fingers[0] == 1:  # Thumb is extended
                        color_selection_frames += 1
                        # Visual feedback: highlight selected color box
                        x1, y1, x2, y2 = box_index * 80, 0, (box_index + 1) * 80, 60
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                        if color_selection_frames >= COLOR_SELECTION_THRESHOLD:
                            candidate_color = colors[box_index]
                            print(f"Selected color box {box_index} at x={tx}, y={ty}")
                    else:
                        color_selection_frames = 0
                else:
                    color_selection_frames = 0
            else:
                color_selection_frames = 0

            # Drawing mode: only index finger up
            if fingers[1] == 1 and sum(fingers[1:]) == 1:
                point_history.append((ix, iy))
                # Smooth coordinates
                if len(point_history) >= 3:
                    avg_x = int(sum(p[0] for p in point_history) / len(point_history))
                    avg_y = int(sum(p[1] for p in point_history) / len(point_history))
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = avg_x, avg_y
                    cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), draw_color, 5)
                    prev_x, prev_y = avg_x, avg_y
            else:
                prev_x, prev_y = 0, 0
                point_history.clear()

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Apply candidate color
        if candidate_color and candidate_color != draw_color:
            draw_color = candidate_color
            color_changed_this_frame = True
            color_selection_frames = 0

        # Clear canvas if both hands are fists
        if len(result.multi_hand_landmarks) == 2:
            if all(is_fist(hand) for hand in result.multi_hand_landmarks):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                print("🧼 Canvas cleared!")

    if color_changed_this_frame and not color_selected:
        play_ding()
        print(f"🎨 Color changed to: {draw_color}")
        color_selected = True
    elif not color_changed_this_frame:
        color_selected = False

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])
    inv_mask_3ch = cv2.bitwise_not(mask_3ch)
    canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    frame_bg = cv2.bitwise_and(frame, inv_mask_3ch)
    canvas_fg = cv2.bitwise_and(canvas, mask_3ch)
    output = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("🖐️ Gesture Drawing - Press Q to Quit", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()