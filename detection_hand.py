import cv2
from hand_tracking_module import HandDetector
import numpy as np
import math
from classificationModule import Classifier 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('model/best_model.h5')

offset = 20
imgSize = 244

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K', 'L', 'M', 'N', 'O', 'P','Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y']
while True:
    success, img = cap.read()
    imgOutPut = img.copy()
    hands, img= detector.findHands(img,handConnection=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        try:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)


            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGal = math.ceil((imgSize-hCal)/2)
                imgWhite[hGal:hCal + hGal,:] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
        except:
            continue

        cv2.rectangle(imgOutPut,(x-offset,y-offset-50),(x +90-offset,y-offset),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutPut,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7 ,(255,255,255),2)
        cv2.rectangle(imgOutPut,(x-offset,y-offset),(x +w+offset,y+h+offset),(255,0,255),4)

    cv2.imshow('Image',imgOutPut)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # Ấn Esc để thoát
            break

