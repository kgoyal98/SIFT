# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:53:54 2019

@author: Aditya Chondke
"""
from siftdetector import detect_keypoints   
import numpy as np
import cv2
import itertools

def match_template1(imagename, templatename, threshold):
    
    img = cv2.imread(imagename)
    template = cv2.imread(templatename)

    [kpi, di] = detect_keypoints(imagename, threshold)
    [kpt, dt] = detect_keypoints(templatename, threshold)
    drawFeatures(imagename, kpi, imagename[:-4]+ "_keypoints.jpg")
    drawFeatures(templatename, kpt, templatename[:-4] + "_keypoints1.jpg")
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:h2,:w2] = template
    newimg[:h1, w2:w1+w2] = img
    matches=0
    for i in range(len(kpi)):
        pt_b = (int(kpi[i,1] + w2), int(kpi[i,0]))
        angles = np.arccos(np.matmul(di[i,:], dt.T))
        indices = range(len(dt))
        # indices.sort(key=lambda i: angles[i])
        indices = [x for _,x in sorted(zip(angles,indices))]
        if(angles[indices[0]]<0.6*angles[indices[1]]):
            pt_a = (int(kpt[indices[0],1]), int(kpt[indices[0],0]))
            cv2.line(newimg, pt_a, pt_b, (255, 255, 255))
            matches+=1
    print ("matched points ", matches)
    cv2.imwrite('matches.jpg', newimg)


def drawFeatures(imagename, keypoints, filename):
    newimg = cv2.imread(imagename)
    r=5.0
    for i in range(len(keypoints)):
        newimg=cv2.circle(newimg, (int(keypoints[i,1]), int(keypoints[i,0])), 2, (255,255,255))
        # newimg = cv2.arrowedLine(newimg, (int(keypoints[i,1]), int(keypoints[i,0])), (int(keypoints[i,1]+r*keypoints[i,4]*np.sin(np.pi/18*keypoints[i,3])), int(keypoints[i,0] +r*keypoints[i,4]*np.cos(np.pi/18*keypoints[i,3]))), (255,255,255), tipLength=0.2)
    cv2.imwrite(filename, newimg)
