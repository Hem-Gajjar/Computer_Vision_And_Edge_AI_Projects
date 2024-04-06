import cv2
import mediapipe as mp
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

fruits = ['banana_small.png', 'apple.png', 'pineapple.png', 'coconut.png', 'orange.png']
fruit_imgs = [cv2.imread(fruit, -1) for fruit in fruits]
fruit_imgs = [cv2.cvtColor(fruit_img, cv2.COLOR_BGRA2BGR) for fruit_img in fruit_imgs]

common_width = 80
fruit_imgs_resized = [cv2.resize(fruit_img, (common_width, int(fruit_img.shape[0] * common_width / fruit_img.shape[1]))) for fruit_img in fruit_imgs]

fruit_states = [{'img': fruit_imgs_resized[i], 'x': random.randint(0, 640), 'y': 0, 'fall_speed': random.randint(1, 5)} for i in range(len(fruits))]

score = 0
miss_count = 0
fall_counter = 0

cam_index = 0
cap = cv2.VideoCapture(cam_index)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_tip = hand_landmarks.landmark[8]
        hand_x = int(hand_tip.x * frame.shape[1])
        hand_y = int(hand_tip.y * frame.shape[0])
        cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)

        for fruit_state in fruit_states:
            fruit_img = fruit_state['img']
            fruit_x = fruit_state['x']
            fruit_y = fruit_state['y']
            if fruit_y + fruit_img.shape[0] >= hand_y >= fruit_y and fruit_x + fruit_img.shape[1] >= hand_x >= fruit_x:
                score += 1
                fruit_state['y'] = -fruit_img.shape[0]
            else:
                if fruit_y >= frame.shape[0]:
                    miss_count += 1
                    fruit_state['y'] = 0
                    fall_counter += 1
                elif fruit_y + fruit_img.shape[0] >= frame.shape[0]:
                    miss_count += 1
                    fall_counter += 1

    for fruit_state in fruit_states:
        fruit_img = fruit_state['img']
        fruit_x = fruit_state['x']
        fruit_y = fruit_state['y']
        if fruit_y >= 0 and fruit_y < frame.shape[0] and fruit_x >= 0 and fruit_x < frame.shape[1]:
            y_start = max(fruit_y, 0)
            y_end = min(fruit_y + fruit_img.shape[0], frame.shape[0])
            x_start = max(fruit_x, 0)
            x_end = min(fruit_x + fruit_img.shape[1], frame.shape[1])
            
            img_y_start = y_start - fruit_y
            img_y_end = img_y_start + (y_end - y_start)
            img_x_start = x_start - fruit_x
            img_x_end = img_x_start + (x_end - x_start)
            
            frame[y_start:y_end, x_start:x_end] = fruit_img[img_y_start:img_y_end, img_x_start:img_x_end]

        fruit_state['y'] += fruit_state['fall_speed']

        if fruit_y > frame.shape[0]:
            fruit_state['y'] = 0
            fall_counter += 1
            miss_count += 1

    if fall_counter >= 500:
        cv2.putText(frame, "Game Over", (150, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Hand Position and Falling Fruits", frame)
        cv2.waitKey(3000)
        break

    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Misses: {miss_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Position and Falling Fruits", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
