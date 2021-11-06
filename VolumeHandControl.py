import os, sys
try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2
try:
    import time
except:
    os.system('pip install time')
    import time
try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np
#################################
import HandTrackingModule as htm
#################################
try:
    import math
except:
    os.syste ('pip install math')
    import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
try: 
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except:
    os.system('pip install pycaw')
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#######################################
# BEGIN HOUSECLEANING
#######################################

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
vol = 0
volBar = 400
volPer = 0
fps = 0

detector = htm.handDetector(detectionCon=1)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

#######################################
# MAIN LOOP
#######################################

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) > 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 -x1, y2 - y1)
        # Hand range 50 - 300
        # Volume Range -65 0
        vol = np.interp(length, [50,300], [minVol, maxVol])
        volBar = np.interp(length, [50,300], [400, 150])
        volPer = np.interp(length, [50,300], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
           cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img,(50,150), (85,400),(0, 255, 0), 3)
    # cv2.rectangle(img,(50,int(volBar)), (85,400),(0, 255, 0), cv2.FILLED)
    cv2.rectangle(img,(55,int(volBar)), (80,395),(0, 0, 162), cv2.FILLED)
    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Vol:{int(volPer)}', (40, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 162), 1)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow('img', img)
#######################################
# ENDING LOGIC
#######################################
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
