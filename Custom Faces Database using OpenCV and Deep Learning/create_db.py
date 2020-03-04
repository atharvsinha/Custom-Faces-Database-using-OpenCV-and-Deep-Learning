from dataDnn import DetectorDnn
from dataHaar import Detector
from pathlib import Path
import os

if __name__ == '__main__':

    #take the name of the person as input
    print("What is the name of the person to be added?")
    name = input()
    
    #lowercase for uniformity
    name = name.lower()
    
    #load the path of the dataset
    dataset = os.path.dirname(os.path.realpath(__file__))
    dataset = os.path.join(dataset, 'dataset')
    ndir = os.path.join(dataset, name)

    #exception handling while creating a new directory in dataset
    try:  
        os.mkdir(ndir)  
    except OSError as error:  
        print(error)

    #pass the directory to the function
    print("For using Haar cascades or OpenCV Deep Learning model Caffe [h/c]")
    option = input()

    if(option == 'H' or option == 'h'):
        Detector(ndir)
    elif(option == 'c' or option == 'C'):
        DetectorDnn(ndir)

