import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller
import time

# --- Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
keyboard = Controller()

# Text storage
finalText = ""

# --- Keyboard Layout ---
# Added "SPACE" and "DEL" for utility
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["SPACE", "DEL"] 
]

COLOR_NEON_CYAN = (255, 255, 0)
COLOR_NEON_GREEN = (0, 255, 0)
COLOR_BG_TRANS = (20, 20, 20)

class Button():
    def __init__(self, pos, text, size=(70, 70)):
        self.pos = pos
        self.size = size
        self.text = text

# Initialize Button Objects
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == "SPACE":
            buttonList.append(Button([100, 400], key, size=(450, 70)))
        elif key == "DEL":
            buttonList.append(Button([580, 400], key, size=(120, 70)))
        else:
            buttonList.append(Button([85 * j + 50, 85 * i + 50], key))

def draw_hud_corners(img, x, y, w, h, color, thickness=2, length=15):
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

def draw_all(img, buttonList, finalText):
    imgNew = np.zeros_like(img, np.uint8)
    
    # Draw the Text Result Box (Top Display)
    cv2.rectangle(imgNew, (50, 490), (700, 580), COLOR_BG_TRANS, cv2.FILLED)
    draw_hud_corners(imgNew, 50, 490, 650, 90, COLOR_NEON_CYAN, 2, 20)
    cv2.putText(imgNew, finalText, (60, 550), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)

    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(imgNew, (x, y), (x + w, y + h), COLOR_BG_TRANS, cv2.FILLED)
        draw_hud_corners(imgNew, x, y, w, h, COLOR_NEON_CYAN, 2, 12)
        cv2.putText(imgNew, button.text, (x + 10, y + 45), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_NEON_CYAN, 2)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, 1 - alpha, imgNew, alpha, 0)[mask]
    return out

# --- Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

last_click_time = 0
click_delay = 0.4

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    img = draw_all(img, buttonList, finalText)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            x8, y8 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
            x12, y12 = int(handLms.landmark[12].x * w), int(handLms.landmark[12].y * h)
            
            # Pointer Circle
            cv2.circle(img, (x8, y8), 10, COLOR_NEON_CYAN, cv2.FILLED)

            for button in buttonList:
                bx, by = button.pos
                bw, bh = button.size
                
                if bx < x8 < bx + bw and by < y8 < by + bh:
                    # Hover effect
                    draw_hud_corners(img, bx, by, bw, bh, COLOR_NEON_GREEN, 4, 20)
                    
                    # Distance between Index (8) and Middle (12)
                    length = math.hypot(x12 - x8, y12 - y8)
                    
                    # --- CLICK ACTION (PINCH GESTURE) ---
                    if length < 40 and (time.time() - last_click_time) > click_delay:
                        if button.text == "SPACE":
                            keyboard.press(" ")
                            finalText += " "
                        elif button.text == "DEL":
                            finalText = finalText[:-1]
                        else:
                            keyboard.press(button.text)
                            finalText += button.text
                        
                        # Click Animation (Flash)
                        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), cv2.FILLED)
                        last_click_time = time.time()
                        
    cv2.imshow("Cyber Keyboard HUD", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()