# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:14:36 2020
this code is from https://medium.com/@somaniswastik/face-recognition-using-tensorflow-pre-trained-model-opencv-91184efa4aaf

"""

import cv2
import numpy as np
import os
from pathlib import Path
import shutil


#collect the images from the directory
def dataset(sourcePicDir):
    images=[]
    labels=[]
    labels_dic={}
   
    i=0
    for person in os.listdir(sourcePicDir):
        labels_dic[i]=Path(person).stem
        personDir = os.path.join(sourcePicDir, person)
        images.append(cv2.imread(personDir,cv2.IMREAD_COLOR))
        labels.append(person)
        i+=1
        
    return(images, np.array(labels),labels_dic)
    


class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        #w_rm = int(0.3 * w / 2)
        #faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
        faces.append(image[y: y + h, x: x + w ])
         
    return faces

def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm



def normalize_faces(image, faces_coord, h_size,v_size):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces, (h_size,v_size))
    
    return faces
  
def crop_faces(sourcePicDir, destPicDir, h_size, v_size):
    
    #if sourcePicDir[:-1] != '\\':
     #   sourcePicDir += '\\'
    #if destPicDir[:-1] != '\\':
     #   destPicDir += '\\'
    if os.path.exists(destPicDir):     
        shutil.rmtree(destPicDir) 
    
    os.makedirs(destPicDir)
    
    images,labels,labels_dic = dataset(sourcePicDir)
  
    for pic_num, image in enumerate(images):
        #detector = FaceDetector("haarcascade_frontalface_default.xml") #miss 1 Toon face and 2 Noon face out of 14 faces 6 pics
        #detector = FaceDetector("haarcascade_frontalface_alt_tree.xml") #only Toon came thru out of 8 faces 3 pics
        detector = FaceDetector("haarcascade_frontalface_alt2.xml") #miss 2 Noon face of 14 faces (Toon Noon Roong) 6 pics
        #detector = FaceDetector("haarcascade_profileface.xml") #doesn't detect anything but 2 background pictures
        faces_coord = detector.detect(image, True)
        faces = normalize_faces(image ,faces_coord, h_size, v_size)
        
        for face_num, face in enumerate(faces):
            outputFilePath = os.path.join(destPicDir,labels_dic[pic_num]+str(face_num)+'.jpg')
            cv2.imwrite(outputFilePath, faces[face_num])
             
    
#crop_faces('C:\\Lu\\AI\\classes\\Tensorflow_ML_DL\\LusTest\\faces_test\\','C:\\Lu\\AI\\classes\\Tensorflow_ML_DL\\LusTest\\faces_test\\output\\')    
#inputDir='C:\\Lu\\AI\\classes\\Tensorflow_ML_DL\\LusTest\\Lu_vs_Kayne\\Cross_validation\\Noon'
#outputDir = os.path.join(inputDir, "cropped")
#crop_faces(inputDir, outputDir)    
