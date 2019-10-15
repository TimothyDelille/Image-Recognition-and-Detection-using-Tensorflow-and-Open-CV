#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:39:20 2019

@author: timothydelille
"""

import cv2
import numpy as np
import random

from libs.ustr import ustr
from libs.labelFile import LabelFile
import os.path
            
import matplotlib.pyplot as plt

def noisy(image, mean=0, var=1000):
    """
    image : ndarray
    Input image data. Will be converted to float.
    mode : str

    Gaussian-distributed additive noise.
    """
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

width = 600
height = 600 

background_herbe = [[10,130,0],
                    [8,99,0],
                    [16,175,1],
                    [5,58,0],
                    [36,137,27],
                    [62,155,54],
                    [75,132,70],
                    [64,96,62],
                    [33,112,22]]

background_uni = [[240,240,240],
                  [40,40,40],
                  [10,10,10],
                  [250,250,250],
                  [0,0,0],
                  [244,66,238],
                  [244,235,66],
                  [176,244,66],
                  [69,244,66],
                  [66,244,179]] 

labels = ['red_line',
          'yellow_line',
          'blue_line',
          'red_cross',
          'yellow_cross',
          'blue_cross']
#BGR
shades_of_red = [(0,0,255),
                 (35,35,220),
                 (50,50,180),
                 (80,80,250),
                 (0,0,200),
                 (240,70,70),
                 (23,23,175)]
shades_of_blue = [(255,0,0),
                  (255,90,90),
                  (200,0,0),
                  (200,50,50),
                  (175,60,60),
                  (175,23,23),
                  (210,85,85)]
shades_of_yellow = [(0,255,255),
                    (130,255,255),
                    (210,255,255),
                    (108,210,210),
                    (0,200,200),
                    (45,230,230),
                    (40,255,255),
                    (165,255,255),
                    (206,255,255)]

#on veut 150 images de rectangles par couleur
cpt = 0
while cpt<=6*1500:
    
    if cpt%6 == 0:
        label_choice = 'red_line'
        color = random.choice(shades_of_red)
    elif cpt%6 == 1:
        label_choice = 'blue_line'
        color = random.choice(shades_of_blue)
    elif cpt%6 == 2:
        label_choice = 'yellow_line'
        color = random.choice(shades_of_yellow)
    elif cpt%6 == 3:
        label_choice = 'red_cross'
        color = random.choice(shades_of_red)
    elif cpt%6 == 4:
        label_choice = 'blue_cross'
        color = random.choice(shades_of_blue)
    elif cpt%6 == 5:
        label_choice = 'yellow_cross'
        color = random.choice(shades_of_yellow)
    
    cpt+=1
    
    img = np.zeros((width, height, 3), dtype = "uint8")

    background_choice = random.choice(('herbe','uni','random'))
    for i in range(len(img)):
        for j in range(len(img[i])):
            if background_choice == 'herbe':
                background_color = random.choice(background_herbe)
                k = random.choice((0,1))
                k = k*np.random.randint(0,50) #inserer des ombres dans l'herbe
                img[i][j] = np.array(background_color, dtype='uint8')+np.array([k,k,k], dtype='uint8')
            elif background_choice == 'uni':
                background_color = random.choice(background_uni)
                img[i][j] = np.array(background_color, dtype='uint8')
            elif background_choice == 'random':
                background_color = random.choice(background_uni)
                img[i][j] = np.array(background_color, dtype='uint8')
    
    if background_choice == 'random':
        img = noisy(img, var=np.random.randint(0,5000))
                
    rows,cols,ch = img.shape
    center = (cols//2, rows//2)

    longueur = np.random.randint(180,220) #longueur du rectangle
    largeur = np.random.randint(30,50) #largeur du rectangle
    diagonale = np.sqrt(longueur**2 + largeur**2)
    
    angle_deg = np.random.randint(-89,89)
    angle_rad = angle_deg*np.pi/180.
    scale = 1
    
    theta = np.arctan(largeur/longueur)
    
    x = int(abs(max(np.cos(angle_rad - theta)*diagonale,np.cos(angle_rad + theta)*diagonale, key=abs)))
    y = int(abs(max(np.sin(angle_rad - theta)*diagonale,np.sin(angle_rad + theta)*diagonale, key=abs)))    
            
    M_rot = cv2.getRotationMatrix2D(center,angle_deg,scale) #center of rot., angle and scale
    
    amplitude = 100
    def_amp = 10
    pt_def1 = np.array([np.random.randint(-def_amp,def_amp),
                           np.random.randint(-def_amp,def_amp)])
    pt_def2 = np.array([np.random.randint(-def_amp,def_amp),
                           np.random.randint(-def_amp,def_amp)])
    pt_def3 = np.array([np.random.randint(-def_amp,def_amp),
                           np.random.randint(-def_amp,def_amp)])
        
    pt1 = np.array([center[0]+amplitude//2,center[1]])
    pt2 = np.array([center[0]-amplitude//2,center[1]-amplitude//2])
    pt3 = np.array([center[0]+amplitude//2,center[1]-amplitude//2])
    
    pt4 = pt1 + pt_def1
    pt5 = pt2 + pt_def2
    pt6 = pt3 + pt_def3
    
    pts1 = np.float32([pt1,pt2,pt3])
    pts2 = np.float32([pt4,pt5,pt6])
    
    M_warp = cv2.getAffineTransform(pts1,pts2)
    #img = noisy(img, var=np.random.randint(0,5000))
    
    points = [(center[0]-longueur//2,center[1]+largeur//2),
              (center[0]+longueur//2,center[1]-largeur//2),
              (center[0]-largeur//2,center[1]+longueur//2),
              (center[0]+largeur//2,center[1]-longueur//2)]    
    
    points_def = np.append(np.array(points).T,[np.ones(len(points))], axis=0)
    points_def = M_rot.dot(points_def)
    points_def = np.append(np.array(points_def),[np.ones(len(points_def.T))], axis=0)
    points_def = M_warp.dot(points_def)
    points_def = points_def.T.astype(int)
    
    shape_center = abs((points_def[1]+points_def[0])/2.)
    shape_center = shape_center.astype(int)
    
    marge = 10 #px
    
            
    points_nbr = 8
    defauts_x = np.linspace(points[0][0],points[1][0],points_nbr)
    defauts_x = np.append(defauts_x,np.flip(defauts_x,0))
    defauts_x = np.append(defauts_x,[points[0][0]])
    defauts_y1 = np.random.randint(points[0][1]-largeur//8,points[0][1]+largeur//8,points_nbr-2)
    defauts_y2 = np.random.randint(points[1][1]-largeur//8,points[1][1]+largeur//8,points_nbr-2)
    defauts_y = np.append(np.array(points[0][1]),defauts_y1)
    defauts_y = np.append(defauts_y,np.array([points[0][1],points[1][1]]))
    defauts_y = np.append(defauts_y,defauts_y2)
    defauts_y = np.append(defauts_y,np.array([points[1][1],points[0][1]]))
    defauts = np.array([defauts_x,defauts_y]).T.astype(int)
    
    if label_choice[-4:]=='line':
        #cv2.rectangle(img, points[0], points[1], color, -1)
        diff = points_def[1]-points_def[0]
        cv2.fillConvexPoly(img,defauts,list(color))        
    
    elif label_choice[-5:]=='cross':        
        #cv2.rectangle(img, points[0], points[1], color, -1)
        #cv2.rectangle(img, points[2], points[3], color, -1)
        
        diff = np.append(points_def[1]-points_def[0], points_def[3]-points_def[2])
        
        defauts_y_cross = np.linspace(points[3][1],points[2][1],points_nbr) #haut gauche, bas gauche, bas droite, haut droite
        defauts_y_cross = np.append(defauts_y_cross,np.flip(defauts_y_cross,0))
        defauts_y_cross = np.append(defauts_y_cross,[points[3][1]])
        defauts_x1_cross = np.random.randint(points[2][0]-largeur//8,points[2][0]+largeur//8,points_nbr-2)
        defauts_x2_cross = np.random.randint(points[3][0]-largeur//8,points[3][0]+largeur//8,points_nbr-2)
        defauts_x_cross = np.append(np.array(points[2][0]),defauts_x1_cross)
        defauts_x_cross = np.append(defauts_x_cross,np.array([points[2][0],points[3][0]]))
        defauts_x_cross = np.append(defauts_x_cross,defauts_x2_cross)
        defauts_x_cross = np.append(defauts_x_cross,[points[3][0],points[2][0]])
        defauts_cross = np.array([defauts_x_cross,defauts_y_cross]).T.astype(int)
        cv2.fillConvexPoly(img,defauts_cross,list(color))
        cv2.fillConvexPoly(img,defauts,list(color)) 
            
    max_dim = int(abs(max(diff, key=abs)))
    x1_box = max(0,shape_center[0]-max_dim//2-marge)
    y1_box = min(rows,shape_center[1]+max_dim//2+marge)
    x2_box = min(cols,shape_center[0]+max_dim//2+marge)
    y2_box = max(0,shape_center[1]-max_dim//2-marge)
    detection_box = [(x1_box,y1_box),(x2_box,y2_box)]
      
    img = cv2.warpAffine(img,M_rot,(cols,rows))
    img = cv2.warpAffine(img,M_warp,(cols,rows))
            
    
    shape = [{'points':detection_box,
              'label':label_choice, 
              'difficult':False, 
              'line_color':None, 
              'fill_color':None}]
    
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #train_dir = os.path.join(dir_path, 'train')
    #test_dir = os.path.join(dir_path, 'test')
    train_dir = './train/'
    test_dir = './test/'
    
    if cpt%60 in [0,1,2,3,4,5]:
        saveDir = test_dir
    else:
        saveDir = train_dir
    
    filename = label_choice+'_'+str(cpt)
    filepath = os.path.join(ustr(saveDir), filename)
    filepath = ustr(filepath)

    cv2.imwrite(filepath+'.jpg',img)
    labelfile = LabelFile()
    labelfile.savePascalVocFormat(filepath+'.xml',shape,filepath+'.jpg',None)
    
    #plt.subplot(121),plt.imshow(img),plt.title('Image')
    #plt.show()

print("Terminé!")