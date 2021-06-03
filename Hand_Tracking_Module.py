import cv2
import mediapipe as mp
import time


class handDetector():
#This class has methods: findHands,findPosition
    def __init__ (self, mode=False,MaxNumOfHands=2,DetectionConfidnc=0.5,TrackConfidnc=0.6):
        self.mode = mode
        self.MaxNumOfHands = MaxNumOfHands
        self.DetectionConfidnc = DetectionConfidnc
        self.TrackConfidnc=TrackConfidnc

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands( self.mode, self.MaxNumOfHands, self.DetectionConfidnc, self.TrackConfidnc)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):    
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: #gives result for all(2) hands
            #handLMs are 21 points. so we need conection too-->mpHands.HAND_CONNECTIONS
                if draw: # if draw=True
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #drawing points and lines(=handconections)
        return img 
    
    def findPosition(self, img, handNo=0, draw=True):
        lmlist=[] #landmark list
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo] #selecting specific hand, != all hand
            for id, lm in enumerate(myHand.landmark):#landamarks for selected hand
                #print(id, lm)
                #lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                #So, need to covert in integer
                h, w, c =img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():
    frameWidth = 1280
    frameHeight = 720
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    pTime = 0
    cTime = 0

    detector=handDetector()

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

if __name__ == "__main__":
    main()