import numpy as np 
import cv2
import os
def DetectorDnn(ndir):
    video = cv2.VideoCapture(0)
    count = 1
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models/deploy.prototxt.txt')
    net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
    while(True):
        _ , img = video.read(0)
        temp_confidence=0
        (h, w) = img.shape[:2]
        temp_img = cv2.resize(img, (w//2, h//2))
        blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.5,(300,300),(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        k = cv2.waitKey(1)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence >.5:
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY), (155, 155, 100), 2)
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                temp_confidence = confidence
            cv2.imshow("Face Detector", img)
        if k%256==27:
            break     
        if k%256==32:
            if temp_confidence <.5:
                print("Not a proper image!")
                continue
            temp = ndir
            img_name = f"{count}.png"
            print(f"{count} images written")
            temp = os.path.join(temp, img_name)            
            cv2.imwrite(temp, temp_img)
            count += 1
    video.release()
    cv2.destroyAllWindows()