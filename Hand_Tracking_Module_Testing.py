import cv2
import mediapipe as mp
import time
import Hand_Tracking_Module as htm

frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
pTime = 0
cTime = 0

detector=htm.handDetector()

while True:
    success, img = cap.read()
    img= cv2.flip(img,1)
    img=detector.findHands(img)
    lmlist= detector.findPosition(img)
    if len(lmlist)!=0:
        print(lmlist[4])

    #Write frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)


    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break