import os, sys
try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2
try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np
try:
    import time
except:
    os.system('pip install time')
    import time
################################
import HandTrackingModule as htm
################################

folderPath = "./Header"
myList = os.listdir(folderPath)
overlayList = []
brushThickness = 15
eraserThickness = 50
defaultColor = (0, 0, 0)
colorPicked  = (255, 0, 255)
blue = (255, 0, 0)
orange = (0,165,255)
lightblue= (255,165,0)
green = (42, 222, 120)
red = (0, 0, 255)
pink = (151, 64, 228)
black = (0, 0, 0)
pTime = 0
fps   = 0

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
#print("header", header)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon = 1)
bbox = []
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)


    #2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw = False)
    if len(lmList) > 0:
        
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which fingers are up

        fingers = detector.fingersUp()
        #print("fingers:", fingers)

    # 3.5 Empty Canvas if thumb and pinky up
        if fingers[0] and fingers[4]:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    #4. If Selection Mode - Two fingers are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 -25), (x2, y2 + 25), (colorPicked), cv2.FILLED)
            print("Selection Mode")
            # Checking for the click
            if y1 < 100:
                if 0 < x1 < 200:
                    colorPicked = blue
                if 200 < x1 < 400:
                    colorPicked = orange
                if 400 < x1 < 600:
                    colorPicked = lightblue
                if 600 < x1 < 800:
                    colorPicked = green
                if 800 < x1 < 1000:
                    colorPicked = red
                if 1000 < x1 < 1150:
                    colorPicked = pink
                if 1150 < x1:
                    colorPicked = black


    #5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (colorPicked), cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if colorPicked == black:
                cv2.line(img, (xp, yp), (x1, y1), colorPicked, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), colorPicked, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), colorPicked, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), colorPicked, brushThickness)
            
            xp, yp = x1, y1
      
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    img[0:100,0:1280] = header
    cv2.putText(img, f'FPS:{int(fps)}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    cv2.imshow("Image", img)
    #cv2.imshow("imgInv", imgInv)
    #cv2.imshow("imageCanvas", imgCanvas)
#######################################
# ENDING LOGIC
#######################################
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
