import os

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2




cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')





imgBackground = cv2.imread('Resources/background.png')
# import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pastList = os.listdir(pathFolderWaste)
for path in pastList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste,path),cv2.IMREAD_UNCHANGED))



while True:
    _, img = cap.read()
    imgResize= cv2.resize(img,(454,340))




    predection = classifier.getPrediction(img)
    print(predection)
    classID = predection[1]





    if predection:
     imgBackground = cvzone.overlayPNG(imgBackground,imgWasteList[classID-1],(909,127))




    imgBackground[148:148+340, 159:159+454] = imgResize
    # Displays
    cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)






