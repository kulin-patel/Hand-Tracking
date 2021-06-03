import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #handLMs are points. so we need conection too-->mpHands.HAND_CONNECTIONS
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #Write frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)

    cv2.imshow('image', img)
    if cv2.waitKey(1)==27:
        break