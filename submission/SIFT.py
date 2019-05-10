# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:48:21 2019

@author: Aditya Chondke
"""


from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import ttk
from tkinter.ttk import Progressbar
from PIL import Image,ImageTk
from tkinter.messagebox import showinfo
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import numpy as np
from scipy import signal
from scipy import misc
from scipy import ndimage
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg
import itertools
from numba import jit,double
import time
import threading




path2=""
path3=""
basefolder=""
progress_bar=""
kpi=0
kpt=0
matches=0


class Window(Frame):
    
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.master=master
        self.init_window()
    
    def init_window(self):
        self.master.title("SIFT")
        self.grid(column=2048, row=2048, sticky=(N, W, E, S))
        
        menu=Menu(self.master)
        self.master.config(menu=menu)
        
        file=Menu(menu)        
        file.add_command(label='Open Image 1',command=self.openImg)
        menu.add_cascade(label='File',menu=file)
        file.add_command(label='Open Image 2',command=self.openImg2)
        
        
     
        
        help1=Menu(menu)
        help1.add_command(label='Change Directory',command=self.changedir)
        menu.add_cascade(label='Help',menu=help1)
        help1.add_command(label='GitHub',command=self.gohelp)
       
        
        
        
        
        button1=Button(self,text="SIFT",command=self.start_submit_thread)
        button1.grid(column=0, row=1)
        button2=Button(self,text="Show Output",command=self.showout)
        button2.grid(column=1, row=1)
      
        
        global progress_bar
        progress_bar=ttk.Progressbar(self,orient='horizontal', length=200,mode='determinate')
        
        
    
        
    def showout(self):
            keypoints2=basefolder+"\\\\keypoints2.jpg"
            keypoints1=basefolder+"\\\\keypoints1.jpg"
            matche=basefolder+"\\\\matches.jpg"
            load2=Image.open(keypoints2)
            load2=load2.resize((256,256),Image.ANTIALIAS)
            render=ImageTk.PhotoImage(load2)
            img2=Label(self,image=render)
            img2.image=render
            img2.grid(column=0,row=3)
            
            load3=Image.open(keypoints1)
            load3=load3.resize((256,256),Image.ANTIALIAS)
            render3=ImageTk.PhotoImage(load3)
            img3=Label(self,image=render3)
            img3.image=render3
            img3.grid(column=1,row=3)
            
            load4=Image.open(matche)
            load4=load4.resize((512,256),Image.ANTIALIAS)
            render4=ImageTk.PhotoImage(load4)
            img4=Label(self,image=render4)
            img4.image=render4
            img4.grid(column=3,row=3)
            
            label1=Label(self,text="Keypoints in Image : "+str(len(kpi))).grid(column=0, row=4 )
            label2=Label(self,text="Keypoints in Template : "+str(len(kpt))).grid(column=0, row=5 )
            label3=Label(self,text="Matched points : "+str(matches)).grid(column=0, row=6 )
            
            

    def gohelp(self):
        webbrowser.open_new('https://github.com/AdityaChondke/SIFT-Algorithm-')
        

        

        
    def start_submit_thread(self):
        global submit_thread
        submit_thread = threading.Thread(target=siftout)
        submit_thread.daemon = True
        progress_bar.grid(column=2,row=1,pady=10)

        progress_bar.start()
        submit_thread.start()
        
    def changedir(self):
        temp=askdirectory()
        m=temp.split('/')
        global basefolder
        basefolder=('\\\\'.join(m))
        wd=('/'.join(m))
        showinfo("Working Directory changed","Working directory set to:  "+wd)
        
        
        
    
    def openImg(self):
        root=Tk()
        root.fileName=filedialog.askopenfilename( filetypes=( ("All files","*.*"),("Image",".jpeg")))
        tpath=root.fileName
        l=tpath.split("/")
        m=l[:-1]

        global path2
        path2=('\\\\'.join(l))
        global basefolder
        basefolder=('\\\\'.join(m))
        wd=('/'.join(m))
        root.destroy()
        load=Image.open(path2)
        load=load.resize((256,256),Image.ANTIALIAS)
        
        render=ImageTk.PhotoImage(load)
        img=Label(self,image=render)
        img.image=render
        img.grid(column=0,row=2)
        
        showinfo("Image Opened","Working directory set to:  "+wd+"\n")
        
    def openImg2(self):
        root=Tk()
        root.fileName=filedialog.askopenfilename( filetypes=( ("All files","*.*"),("Image",".jpeg")))
        tpath=root.fileName
        l=tpath.split("/")
        m=l[:-1]

        global path3
        path3=('\\\\'.join(l))
        global basefolder
        basefolder=('\\\\'.join(m))
        wd=('/'.join(m))
        root.destroy()
        load=Image.open(path3)
        load=load.resize((256,256),Image.ANTIALIAS)
        
        render=ImageTk.PhotoImage(load)
        img=Label(self,image=render)
        img.image=render
        img.grid(column=1,row=2)
        
        showinfo("Image Opened","Working directory set to:  "+wd+"\n")
        
        
        

        
    def client_exit(self):
        exit()
        
def siftout():
    start=time.time()
    
    global kpi
    global kpt
    global matches
    (kpi,kpt,matches)=match_template1(path3,path2 , 5)
    
    
    end=time.time()
    
    progress_bar.stop()
    
    

def dog(pyrlvl1,pyrlvl2,pyrlvl3,pyrlvl4):
    diffpyrlvl1 = np.zeros((pyrlvl1.shape[0], pyrlvl1.shape[1], 5))
    diffpyrlvl2 = np.zeros((pyrlvl2.shape[0], pyrlvl2.shape[1], 5))
    diffpyrlvl3 = np.zeros((pyrlvl3.shape[0], pyrlvl3.shape[1], 5))
    diffpyrlvl4 = np.zeros((pyrlvl4.shape[0], pyrlvl4.shape[1], 5))
    for i in range(0, 5):
        diffpyrlvl1[:,:,i] = pyrlvl1[:,:,i+1] - pyrlvl1[:,:,i]
        diffpyrlvl2[:,:,i] = pyrlvl2[:,:,i+1] - pyrlvl2[:,:,i]
        diffpyrlvl3[:,:,i] = pyrlvl3[:,:,i+1] - pyrlvl3[:,:,i]
        diffpyrlvl4[:,:,i] = pyrlvl4[:,:,i+1] - pyrlvl4[:,:,i]
    return(diffpyrlvl1,diffpyrlvl2,diffpyrlvl3,diffpyrlvl4)


dog_fast = jit(double[:,:](double[:,:],double[:,:],double[:,:],double[:,:]))(dog)

def magnori(pyrlvl1):
     magpyrlvl1 = np.zeros((pyrlvl1.shape[0], pyrlvl1.shape[1], 3))
     oripyrlvl1=np.zeros((pyrlvl1.shape[0], pyrlvl1.shape[1], 3))
     for i in range(0, 3):
        for j in range(1, pyrlvl1.shape[0] - 1):
            for k in range(1, pyrlvl1.shape[1] - 1):
                magpyrlvl1[j, k, i] = ( ((pyrlvl1[j+1, k,i] - pyrlvl1[j-1, k,i]) ** 2) + ((pyrlvl1[j, k+1,i] - pyrlvl1[j, k-1,i]) ** 2) ) ** 0.5   
                oripyrlvl1[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((pyrlvl1[j, k+1,i] - pyrlvl1[j, k-1,i]), (pyrlvl1[j+1, k,i] - pyrlvl1[j-1, k,i])))        
     return(magpyrlvl1,oripyrlvl1)


magnori_fast = jit(double[:,:](double[:,:]))(magnori)


def detect_keypoints(imagename, threshold):
    original = cv2.imread(imagename,0).astype(float)
    

    # SIFT Parameters
    s = 3
    k = 2 ** (1.0 / s)
    # threshold variable is the contrast threshold. Set to at least 1

    # Standard deviations for Gaussian smoothing
    kvec1 = np.array([1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5)])
    kvec2 = np.array([1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8)])
    kvec3 = np.array([1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])
    kvec4 = np.array([1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13), 1.6 * (k ** 14)])
    kvectotal = np.array([1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])

    # Downsampling images
    doubled = misc.imresize(original, 200, 'bilinear').astype(float)
    # doubled = bilinear2(original)
    doubled = ndimage.filters.gaussian_filter(doubled, np.sqrt(1.6**2-0.5**2*4)).astype(float)
    normal = misc.imresize(doubled, 50, 'bilinear').astype(float)
    halved = misc.imresize(normal, 50, 'bilinear').astype(float)
    quartered = misc.imresize(halved, 50, 'bilinear').astype(float)

    # Initialize Gaussian pyramids
    pyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 6))
    pyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 6))
    pyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 6))
    pyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 6))

    print ("Constructing pyramids...")

    # Construct Gaussian pyramids
    for i in range(0, 6):
        pyrlvl1[:,:,i] = ndimage.filters.gaussian_filter(doubled, kvec1[i])   
        pyrlvl2[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec2[i]), 50, 'bilinear') 
        pyrlvl3[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec3[i]), 25, 'bilinear')
        pyrlvl4[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec4[i]), 1.0 / 8.0, 'bilinear')

    # Initialize Difference-of-Gaussians (DoG) pyramids
    diffpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    diffpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 5))
    diffpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 5))
    diffpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 5))

    # Construct DoG pyramids
    
    (diffpyrlvl1,diffpyrlvl2,diffpyrlvl3,diffpyrlvl4)=dog_fast(pyrlvl1,pyrlvl2,pyrlvl3,pyrlvl4)
    
    

    # Initialize pyramids to store extrema locations
    extrpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 3)).astype(int)
    extrpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 3)).astype(int)
    extrpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 3)).astype(int)
    extrpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 3)).astype(int)

    print ("Starting extrema detection...")
    print ("First octave")

    
    
    
    for i in range(1, 4):
        for j in range(20, doubled.shape[0] - 20):
            for k in range(20, doubled.shape[1] - 20):
                if np.absolute(diffpyrlvl1[j, k, i]) < threshold:
                    continue        

                maxbool = (diffpyrlvl1[j, k, i] > 0)
                minbool = (diffpyrlvl1[j, k, i] < 0)


                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (diffpyrlvl1[j, k, i] > diffpyrlvl1[j + dj, k + dk, i + di])
                            minbool = minbool and (diffpyrlvl1[j, k, i] < diffpyrlvl1[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break

                        if not maxbool and not minbool:
                            break

                    if not maxbool and not minbool:
                        break
                
                if maxbool or minbool:
                    dx = (diffpyrlvl1[j, k+1, i] - diffpyrlvl1[j, k-1, i]) * 0.5 / 255
                    dy = (diffpyrlvl1[j+1, k, i] - diffpyrlvl1[j-1, k, i]) * 0.5 / 255
                    ds = (diffpyrlvl1[j, k, i+1] - diffpyrlvl1[j, k, i-1]) * 0.5 / 255
                    dxx = (diffpyrlvl1[j, k+1, i] + diffpyrlvl1[j, k-1, i] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255        
                    dyy = (diffpyrlvl1[j+1, k, i] + diffpyrlvl1[j-1, k, i] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255          
                    dss = (diffpyrlvl1[j, k, i+1] + diffpyrlvl1[j, k, i-1] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255
                    dxy = (diffpyrlvl1[j+1, k+1, i] - diffpyrlvl1[j+1, k-1, i] - diffpyrlvl1[j-1, k+1, i] + diffpyrlvl1[j-1, k-1, i]) * 0.25 / 255 
                    dxs = (diffpyrlvl1[j, k+1, i+1] - diffpyrlvl1[j, k-1, i+1] - diffpyrlvl1[j, k+1, i-1] + diffpyrlvl1[j, k-1, i-1]) * 0.25 / 255 
                    dys = (diffpyrlvl1[j+1, k, i+1] - diffpyrlvl1[j-1, k, i+1] - diffpyrlvl1[j+1, k, i-1] + diffpyrlvl1[j-1, k, i-1]) * 0.25 / 255  
                    
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = numpy.linalg.lstsq(H, dD)[0]
                    D_x_hat = diffpyrlvl1[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)
                 
                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))):# and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
                        extrpyrlvl1[j, k, i - 1] = 1

    print ("Second octave")

    for i in range(1, 4):
        for j in range(20, normal.shape[0] - 20):
            for k in range(20, normal.shape[1] - 20):
                if np.absolute(diffpyrlvl2[j, k, i]) < threshold:
                    continue        

                maxbool = (diffpyrlvl1[j, k, i] > 0)
                minbool = (diffpyrlvl1[j, k, i] < 0)

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (diffpyrlvl2[j, k, i] > diffpyrlvl2[j + dj, k + dk, i + di])
                            minbool = minbool and (diffpyrlvl2[j, k, i] < diffpyrlvl2[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break

                        if not maxbool and not minbool:
                            break

                    if not maxbool and not minbool:
                        break
                
                if maxbool or minbool:
                    dx = (diffpyrlvl2[j, k+1, i] - diffpyrlvl2[j, k-1, i]) * 0.5 / 255
                    dy = (diffpyrlvl2[j+1, k, i] - diffpyrlvl2[j-1, k, i]) * 0.5 / 255
                    ds = (diffpyrlvl2[j, k, i+1] - diffpyrlvl2[j, k, i-1]) * 0.5 / 255
                    dxx = (diffpyrlvl2[j, k+1, i] + diffpyrlvl2[j, k-1, i] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255        
                    dyy = (diffpyrlvl2[j+1, k, i] + diffpyrlvl2[j-1, k, i] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255          
                    dss = (diffpyrlvl2[j, k, i+1] + diffpyrlvl2[j, k, i-1] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255
                    dxy = (diffpyrlvl2[j+1, k+1, i] - diffpyrlvl2[j+1, k-1, i] - diffpyrlvl2[j-1, k+1, i] + diffpyrlvl2[j-1, k-1, i]) * 0.25 / 255 
                    dxs = (diffpyrlvl2[j, k+1, i+1] - diffpyrlvl2[j, k-1, i+1] - diffpyrlvl2[j, k+1, i-1] + diffpyrlvl2[j, k-1, i-1]) * 0.25 / 255 
                    dys = (diffpyrlvl2[j+1, k, i+1] - diffpyrlvl2[j-1, k, i+1] - diffpyrlvl2[j+1, k, i-1] + diffpyrlvl2[j-1, k, i-1]) * 0.25 / 255  
                    
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = numpy.linalg.lstsq(H, dD)[0]
                    D_x_hat = diffpyrlvl2[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))):# and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
                        extrpyrlvl2[j, k, i - 1] = 1

    print ("Third octave")
      
    for i in range(1, 4):
        for j in range(20, halved.shape[0] - 20):
            for k in range(20, halved.shape[1] - 20):
                if np.absolute(diffpyrlvl3[j, k, i]) < threshold:
                    continue        

                maxbool = (diffpyrlvl1[j, k, i] > 0)
                minbool = (diffpyrlvl1[j, k, i] < 0)

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (diffpyrlvl3[j, k, i] > diffpyrlvl3[j + dj, k + dk, i + di])
                            minbool = minbool and (diffpyrlvl3[j, k, i] < diffpyrlvl3[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break

                        if not maxbool and not minbool:
                            break

                    if not maxbool and not minbool:
                        break
                
                if maxbool or minbool:
                    dx = (diffpyrlvl3[j, k+1, i] - diffpyrlvl3[j, k-1, i]) * 0.5 / 255
                    dy = (diffpyrlvl3[j+1, k, i] - diffpyrlvl3[j-1, k, i]) * 0.5 / 255
                    ds = (diffpyrlvl3[j, k, i+1] - diffpyrlvl3[j, k, i-1]) * 0.5 / 255
                    dxx = (diffpyrlvl3[j, k+1, i] + diffpyrlvl3[j, k-1, i] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255        
                    dyy = (diffpyrlvl3[j+1, k, i] + diffpyrlvl3[j-1, k, i] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255          
                    dss = (diffpyrlvl3[j, k, i+1] + diffpyrlvl3[j, k, i-1] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255
                    dxy = (diffpyrlvl3[j+1, k+1, i] - diffpyrlvl3[j+1, k-1, i] - diffpyrlvl3[j-1, k+1, i] + diffpyrlvl3[j-1, k-1, i]) * 0.25 / 255 
                    dxs = (diffpyrlvl3[j, k+1, i+1] - diffpyrlvl3[j, k-1, i+1] - diffpyrlvl3[j, k+1, i-1] + diffpyrlvl3[j, k-1, i-1]) * 0.25 / 255 
                    dys = (diffpyrlvl3[j+1, k, i+1] - diffpyrlvl3[j-1, k, i+1] - diffpyrlvl3[j+1, k, i-1] + diffpyrlvl3[j-1, k, i-1]) * 0.25 / 255  
                    
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = numpy.linalg.lstsq(H, dD)[0]
                    D_x_hat = diffpyrlvl3[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))):# and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
                        extrpyrlvl3[j, k, i - 1] = 1
                    
                      
    print ("Fourth octave")

    for i in range(1, 4):
        for j in range(10, quartered.shape[0] - 10):
            for k in range(10, quartered.shape[1] - 10):
                if np.absolute(diffpyrlvl4[j, k, i]) < threshold:
                    continue        

                maxbool = (diffpyrlvl1[j, k, i] > 0)
                minbool = (diffpyrlvl1[j, k, i] < 0)

                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (diffpyrlvl4[j, k, i] > diffpyrlvl4[j + dj, k + dk, i + di])
                            minbool = minbool and (diffpyrlvl4[j, k, i] < diffpyrlvl4[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break

                        if not maxbool and not minbool:
                            break

                    if not maxbool and not minbool:
                        break
                
                if maxbool or minbool:
                    dx = (diffpyrlvl4[j, k+1, i] - diffpyrlvl4[j, k-1, i]) * 0.5 / 255
                    dy = (diffpyrlvl4[j+1, k, i] - diffpyrlvl4[j-1, k, i]) * 0.5 / 255
                    ds = (diffpyrlvl4[j, k, i+1] - diffpyrlvl4[j, k, i-1]) * 0.5 / 255
                    dxx = (diffpyrlvl4[j, k+1, i] + diffpyrlvl4[j, k-1, i] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255        
                    dyy = (diffpyrlvl4[j+1, k, i] + diffpyrlvl4[j-1, k, i] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255          
                    dss = (diffpyrlvl4[j, k, i+1] + diffpyrlvl4[j, k, i-1] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255
                    dxy = (diffpyrlvl4[j+1, k+1, i] - diffpyrlvl4[j+1, k-1, i] - diffpyrlvl4[j-1, k+1, i] + diffpyrlvl4[j-1, k-1, i]) * 0.25 / 255 
                    dxs = (diffpyrlvl4[j, k+1, i+1] - diffpyrlvl4[j, k-1, i+1] - diffpyrlvl4[j, k+1, i-1] + diffpyrlvl4[j, k-1, i-1]) * 0.25 / 255 
                    dys = (diffpyrlvl4[j+1, k, i+1] - diffpyrlvl4[j-1, k, i+1] - diffpyrlvl4[j+1, k, i-1] + diffpyrlvl4[j-1, k, i-1]) * 0.25 / 255  
                    
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = numpy.linalg.lstsq(H, dD)[0]
                    D_x_hat = diffpyrlvl4[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))):# and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
                        extrpyrlvl4[j, k, i - 1] = 1
                     
              
    print ("Number of extrema in first octave: %d" % np.sum(extrpyrlvl1))
    print ("Number of extrema in second octave: %d" % np.sum(extrpyrlvl2))
    print ("Number of extrema in third octave: %d" % np.sum(extrpyrlvl3))
    print ("Number of extrema in fourth octave: %d" % np.sum(extrpyrlvl4))
    
    # Gradient magnitude and orientation for each image sample point at each scale
    magpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    magpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    magpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 3))
    magpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    oripyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    oripyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    oripyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 3))
    oripyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 3))
    
    
    
    
    (magpyrlvl1,oripyrlvl1)=magnori_fast(pyrlvl1)
    (magpyrlvl2,oripyrlvl2)=magnori_fast(pyrlvl2)
    (magpyrlvl3,oripyrlvl3)=magnori_fast(pyrlvl3)
    (magpyrlvl4,oripyrlvl4)=magnori_fast(pyrlvl4)
    
    
    extr_sum = np.sum(extrpyrlvl1) + np.sum(extrpyrlvl2) + np.sum(extrpyrlvl3) + np.sum(extrpyrlvl4)

    keypoints = np.zeros((0, 5)) 

    print ("Calculating keypoint orientations...")

    
    for i in range(0, 3):
        for j in range(10, doubled.shape[0] - 10):
            for k in range(10, doubled.shape[1] - 10):
                if extrpyrlvl1[j, k, i] == 1:
                    sd = 1.5 * kvectotal[i]
                    xlim = int(np.round(3 * sd))
                    orient_hist = np.zeros([36,1])
                    for x in range(-1*xlim, xlim + 1):
                        ylim = int(((xlim** 2) - (x ** 2)) ** 0.5)
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > doubled.shape[0] - 1 or k + y < 0 or k + y > doubled.shape[1] - 1:
                                continue
                            weight = magpyrlvl1[j + x, k + y, i] * gaussian2d(x,y,sd)
                            bin_idx = np.clip(np.floor(oripyrlvl1[j + x, k + y, i]), 0, 35)
                            orient_hist[int(np.floor(bin_idx))] += weight  
                    for _ in range(2):
                        orient_hist=smoothOriHist(orient_hist)
                    maxval = np.amax(orient_hist)
                    for i1 in range(36):
                        l=(i1-1+36)%36
                        r=(i1+1)%36
                        if ( orient_hist[i1] > orient_hist[l] and orient_hist[i1] > orient_hist[r] and orient_hist[i1] >= 0.8*maxval):
                            bin1 = (int(np.round(i1 + 0.5*(orient_hist[l]-orient_hist[r])/(orient_hist[l]-2*orient_hist[i1]+orient_hist[r])))+36)%36
                            keypoints=np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), kvectotal[i], bin1, orient_hist[bin1]]]), axis=0)
    
   
    for i in range(0, 3):
        for j in range(10, normal.shape[0] - 10):
            for k in range(10, normal.shape[1] - 10):
                if extrpyrlvl2[j, k, i] == 1:
                    sd = 1.5 * kvectotal[i+3]
                    xlim = int(np.round(3 * sd))
                    orient_hist = np.zeros([36,1])
                    for x in range(-1*xlim, xlim + 1):
                        ylim = int(((xlim** 2) - (x ** 2)) ** 0.5)
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > normal.shape[0] - 1 or k + y < 0 or k + y > normal.shape[1] - 1:
                                continue
                            weight = magpyrlvl2[j + x, k + y, i] * gaussian2d(x,y,sd)
                            bin_idx = np.clip(np.floor(oripyrlvl2[j + x, k + y, i]), 0, 35)
                            orient_hist[int(np.floor(bin_idx))] += weight  
                    for _ in range(2):
                        orient_hist=smoothOriHist(orient_hist)
                    maxval = np.amax(orient_hist)
                    for i1 in range(36):
                        l=(i1-1+36)%36
                        r=(i1+1)%36
                        if ( orient_hist[i1] > orient_hist[l] and orient_hist[i1] > orient_hist[r] and orient_hist[i1] >= 0.8*maxval ):
                            bin1 = (int(np.round(i1 + 0.5*(orient_hist[l]-orient_hist[r])/(orient_hist[l]-2*orient_hist[i1]+orient_hist[r])))+36)%36
                            keypoints=np.append(keypoints, np.array([[int(j), int(k), kvectotal[i+3], bin1, orient_hist[bin1]]]), axis=0)
    

    for i in range(0, 3):
        for j in range(10, halved.shape[0] - 10):
            for k in range(10, halved.shape[1] - 10):
                if extrpyrlvl3[j, k, i] == 1:
                    sd = 1.5 * kvectotal[i+6]
                    xlim = int(np.round(3 * sd))
                    orient_hist = np.zeros([36,1])
                    for x in range(-1*xlim, xlim + 1):
                        ylim = int(((xlim** 2) - (x ** 2)) ** 0.5)
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > halved.shape[0] - 1 or k + y < 0 or k + y > halved.shape[1] - 1:
                                continue
                            weight = magpyrlvl3[j + x, k + y, i] * gaussian2d(x,y,sd)
                            bin_idx = np.clip(np.floor(oripyrlvl3[j + x, k + y, i]), 0, 35)
                            orient_hist[int(np.floor(bin_idx))] += weight  
                    for _ in range(2):
                        orient_hist=smoothOriHist(orient_hist)
                    maxval = np.amax(orient_hist)
                    for i1 in range(36):
                        l=(i1-1+36)%36
                        r=(i1+1)%36
                        if ( orient_hist[i1] > orient_hist[l] and orient_hist[i1] > orient_hist[r] and orient_hist[i1] >= 0.8*maxval ):
                            bin1 = (int(np.round(i1 + 0.5*(orient_hist[l]-orient_hist[r])/(orient_hist[l]-2*orient_hist[i1]+orient_hist[r])))+36)%36
                            keypoints=np.append(keypoints, np.array([[int(j * 2), int(k * 2), kvectotal[i+6], bin1, orient_hist[bin1]]]), axis=0)
    

    for i in range(0, 3):
        for j in range(10, quartered.shape[0] - 10):
            for k in range(10, quartered.shape[1] - 10):
                if extrpyrlvl4[j, k, i] == 1:
                    sd = 1.5 * kvectotal[i+9]
                    xlim = int(np.round(3 * sd))
                    orient_hist = np.zeros([36,1])
                    for x in range(-1*xlim, xlim + 1):
                        ylim = int(((xlim** 2) - (x ** 2)) ** 0.5)
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > quartered.shape[0] - 1 or k + y < 0 or k + y > quartered.shape[1] - 1:
                                continue
                            weight = magpyrlvl4[j + x, k + y, i] * gaussian2d(x,y,sd)
                            bin_idx = np.clip(np.floor(oripyrlvl4[j + x, k + y, i]), 0, 35)
                            orient_hist[int(np.floor(bin_idx))] += weight  
                    for _ in range(2):
                        orient_hist=smoothOriHist(orient_hist)
                    maxval = np.amax(orient_hist)
                    for i1 in range(36):
                        l=(i1-1+36)%36
                        r=(i1+1)%36
                        if ( orient_hist[i1] > orient_hist[l] and orient_hist[i1] > orient_hist[r] and orient_hist[i1] >= 0.8*maxval ):
                            bin1 = (int(np.round(i1 + 0.5*(orient_hist[l]-orient_hist[r])/(orient_hist[l]-2*orient_hist[i1]+orient_hist[r])))+36)%36
                            keypoints = np.append(keypoints, np.array([[int(j * 4), int(k * 4), kvectotal[i+9], bin1, orient_hist[bin1]]]), axis=0)
    
    
    print ("total keypoints", len(keypoints))
    print ("Calculating descriptor...")
    magpyr = np.zeros((normal.shape[0], normal.shape[1], 12))
    oripyr = np.zeros((normal.shape[0], normal.shape[1], 12))

    for i in range(0, 3):
        magpyr[:, :, i] = misc.imresize(magpyrlvl1[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")   
        oripyr[:, :, i] = misc.imresize(oripyrlvl1[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")  

    for i in range(0, 3):
        magpyr[:, :, i+3] = (magpyrlvl2[:, :, i])
        oripyr[:, :, i+3] = (oripyrlvl2[:, :, i])        
    
    for i in range(0, 3):
        magpyr[:, :, i+6] = misc.imresize(magpyrlvl3[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")
        oripyr[:, :, i+6] = misc.imresize(oripyrlvl3[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")

    for i in range(0, 3):
        magpyr[:, :, i+9] = misc.imresize(magpyrlvl4[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")
        oripyr[:, :, i+9] = misc.imresize(oripyrlvl4[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear")
        

    descriptors = np.zeros([keypoints.shape[0], 128])

    for i in range(0, keypoints.shape[0]): 
        for x in range(-8, 8):
            for y in range(-8, 8):
                theta = 10 * keypoints[i,3] * np.pi / 180.0
                xrot = np.cos(theta) * x - np.sin(theta) * y
                yrot = np.sin(theta) * x + np.cos(theta) * y
                scale_idx = int(np.argwhere(kvectotal == keypoints[i,2])[0][0])
                x0 = int(round(keypoints[i,0]+xrot))
                y0 = int(round(keypoints[i,1]+yrot))
                if(x0<0 or y0<0 or x0>=normal.shape[0] or y0>=normal.shape[1]):
                        continue
                weight = magpyr[x0, y0, scale_idx] * gaussian2d(x, y, 8)
                angle = oripyr[x0, y0, scale_idx] - keypoints[i,3]
                if angle < 0:
                    angle = 36 + angle

                bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
                descriptors[i, 32 * int((x + 8)/4) + 8 * int((y + 8)/4) + bin_idx] += weight
        
        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :]) 
        descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
                
 
    return [keypoints, descriptors]


def gaussian2d(x,y,s):
        return 1.0/(2*np.pi*s**2)*np.exp(-1.0*(x**2+y**2)/2/s**2)

def bilinear2(img):
        (r,c) = img.shape
        i=np.zeros((2*r, 2*c)).astype(float)
        i[::2,::2]= img
        i[1::2,::2]= i[::2,::2]/2
        i[1:-1:2,::2]+= i[2::2,::2]/2
        i[::,1::2]= i[::,::2]/2
        i[::,1:-1:2]+= i[::,2::2]/2
        i[-1:,:]=i[-2:-1,:]
        i[:,-1:]=i[:,-2:-1]
        return i


def smoothOriHist(hist):
    n=hist.shape[0]
    for i in range(n):
        if (i==0):
            prev = hist[n-1,0]
            next = hist[1,0]
        elif (i==n-1):
            prev = hist[i-1,0]
            next = hist[0,0]
        else:
            prev = hist[i-1,0]
            next = hist[i+1,0]
        hist[i,0] = 0.15*prev + 0.7*hist[i,0] + 0.15*next
    return hist


def match_template1(imagename, templatename, threshold):
    keypoints2=basefolder+"\\\\keypoints2.jpg"
    keypoints1=basefolder+"\\\\keypoints1.jpg"
    matche=basefolder+"\\\\matches.jpg"
            
    img = cv2.imread(imagename)
    template = cv2.imread(templatename)

    [kpi, di] = detect_keypoints(imagename, threshold)
    [kpt, dt] = detect_keypoints(templatename, threshold)
    drawFeatures(imagename, kpi, keypoints1)
    drawFeatures(templatename, kpt, keypoints2)
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
    cv2.imwrite(matche, newimg)
    return(kpi,kpt,matches)

def drawFeatures(imagename, keypoints, filename):
    newimg = cv2.imread(imagename)
    r=5.0
    for i in range(len(keypoints)):
        newimg=cv2.circle(newimg, (int(keypoints[i,1]), int(keypoints[i,0])), 2, (255,255,255))
        # newimg = cv2.arrowedLine(newimg, (int(keypoints[i,1]), int(keypoints[i,0])), (int(keypoints[i,1]+r*keypoints[i,4]*np.sin(np.pi/18*keypoints[i,3])), int(keypoints[i,0] +r*keypoints[i,4]*np.cos(np.pi/18*keypoints[i,3]))), (255,255,255), tipLength=0.2)
    cv2.imwrite(filename, newimg)
    
            
        
        
root=Tk()
ment1=StringVar()
ment2=StringVar()
ment3=StringVar()

root.wm_geometry("2048x2048")
                                                                               
app=Window(root)

root.mainloop()

