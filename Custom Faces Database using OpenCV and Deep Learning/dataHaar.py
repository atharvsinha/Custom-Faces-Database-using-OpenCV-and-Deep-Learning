import os
import cv2
from matplotlib import pyplot as plt 
def Detector(ndir):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)
    count = 1
    while(True):
        _ , img = video.read(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(1)
        face = cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(5, 5), maxSize=None)
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (200, 155, 200), 2)
        cv2.imshow("Face Detector", img)
        if k%256==27:
            break 
        if k%256==32:
            temp = ndir
            img_name = f"{count}.png"
            print(f"{count} images written")
            temp = os.path.join(temp, img_name)            
            cv2.imwrite(temp, img)
            count += 1
    video.release()
    cv2.destroyAllWindows()