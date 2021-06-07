import cv2
import mediapipe as mp
import time
import Hand_Tracking_Module as htm
import numpy as np
import urllib.request

url = 'http://192.168.1.17:8080/shot.jpg'
frameWidth = 1280
frameHeight = 720
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

#Saving file
result = cv2.VideoWriter('test_output.avi',cv2.VideoWriter_fourcc(*'XVID'), 20,(frameWidth,frameHeight))


pTime = 0
cTime = 0

detector=htm.handDetector()

while True:
    imgResp = urllib.request.urlopen (url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img=cv2.resize(img,(frameWidth,frameHeight))
    img= cv2.flip(img,1)

    img=detector.findHands(img)
    lmlist= detector.findPosition(img,draw=False)
    # if len(lmlist)!=0:
    #     print(lmlist[4])

    #Write frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)

    result.write(img)
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break