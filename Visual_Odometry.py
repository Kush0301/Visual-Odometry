#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog
import tkinter


# In[3]:


num_of_frames=148
root_dir_path = os.path.dirname(os.path.abspath("__file__"))

#Change directory here
image_dir = os.path.join(root_dir_path, r'/home/kushagra/github/visual_odometry-1/src/KITTI_sample/images/')
images=[]
a='0'
for i in range(num_of_frames+1):
    #Change file name accordingly
    im_name = str(image_dir)+str(a*(6-len(str(i)))+str(i))+".png".format(image_dir)
    images.append(cv2.imread(im_name))
root=tkinter.Tk()
gtdir = filedialog.askopenfilename()
ground_truth = np.loadtxt(gtdir)
root.destroy()


# In[4]:


#Calibration Matrix
k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


# In[21]:


def extract_features(img):
    
    #Using Clahe for better contrast, thus increasing the number of features detected
#     clahe = cv2.createCLAHE(clipLimit=25.0)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img=clahe.apply(img)
    
    #Using FAST
    fast= cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
    kp = fast.detect(img)
    kp = np.array([kp[idx].pt for idx in range(len(kp))], dtype = np.float32)
    return kp


# In[22]:



def track_features(image_ref, image_cur,ref):
    #Initializing LK parameters
    lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref, None, **lk_params)
    
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)
#     distance=abs(ref-kp1).max(-1)

    return kp1, kp2 
def RelativeScale(last_cloud, new_cloud):
    min_idx = min([new_cloud.shape[0],last_cloud.shape[0]])
    p_Xk = new_cloud[:min_idx]
    Xk = np.roll(p_Xk,shift = -3)
    p_Xk_1 = last_cloud[:min_idx]
    Xk_1 = np.roll(p_Xk_1,shift = -3)
    d_ratio = (np.linalg.norm(p_Xk_1 - Xk_1,axis = -1))/(np.linalg.norm(p_Xk - Xk,axis = -1))

    return np.median(d_ratio)


def triangulation(R, t, kp0, kp1, K):
    P0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P0 = K.dot(P0)
    P1 = np.hstack((R, t))
    P1 = K.dot(P1)
    kp0_3d=np.ones((3,kp0.shape[0]))
    kp1_3d=np.ones((3,kp1.shape[0]))
    kp0_3d[0], kp0_3d[1] = kp0[:, 0].copy(), kp0[:, 1].copy()
    kp1_3d[0], kp1_3d[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    cloud = cv2.triangulatePoints(P0, P1, kp0_3d[:2],kp1_3d[:2])
#     cloud/=cloud[3]
    cloud=cloud.T
    return cloud[:,:3]


# In[29]:


trajectory=[]
threshold=1000
#Compute the keypoints for the first set
image_gray_1=cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
kp1 = extract_features(images[0])

image_gray_2=cv2.cvtColor(images[1],cv2.COLOR_BGR2GRAY)
kp2 = extract_features(images[1])

#Use LKT to track the features
kp1,kp2=track_features(image_gray_1, image_gray_2, kp2)

#Calculate Essesntial matrix
E,mask=cv2.findEssentialMat(kp2,kp1,k,cv2.RANSAC, prob=0.999,mask=None)
kp1=kp1[mask.ravel()==1]
kp2=kp2[mask.ravel()==1]

#Obtain rotation and translation for the essential matrix
retval,rmat,trans,mask=cv2.recoverPose(E,kp1,kp2,k)

#Initialize rotation and translation with the first reading
translation = rmat.dot(trans)
rotation = rmat
trajectory.append(translation)
#Compute the cloud to calculate the scale
new_cloud=triangulation(rmat,trans,kp1,kp2,k)
i=1

while(i<=len(images)-1):
    
    image_gray_1=image_gray_2
    old_cloud=new_cloud
    kp1=kp2
    image_gray_2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)


    #Track features
    kp1,kp2=track_features(image_gray_1, image_gray_2, kp1)
    
    #If the number of features tracked falls below 20 then recompute the keypoints
    if kp1.shape[0]<threshold:
        kp2=extract_features(images[i])
        i=i+1
        continue
        
    #Essential Matrix    
    E,mask=cv2.findEssentialMat(kp2,kp1,k,cv2.RANSAC,prob=0.999, mask=None)
    kp1=kp1[mask.ravel()==1]
    kp2=kp2[mask.ravel()==1]
    
  
    
    #Recover translation and rotation
    retval,rmat,trans,mask=cv2.recoverPose(E,kp1,kp2,k)
    
    #Calculate the cloud of the next set
    new_cloud=triangulation(rmat,trans,kp1,kp2,k)
    
    #Compare the two clouds to recover the scale factor
    scale= -RelativeScale(old_cloud, new_cloud)
    #Propagate translation and rotation
    translation=translation+scale*rotation.dot(trans)
    rotation=rotation.dot(rmat)
    trajectory.append(translation)
   

    #If the number of features tracked falls below 20 then recompute the keypoints
    if kp1.shape[0]<threshold:
        kp2=extract_features(images[i])
    
        
    i=i+1


# In[31]:


trajectory=np.array(trajectory)
x,y,z=[],[],[]
for i in range(0, trajectory.shape[0]):
    x.append(trajectory[i,0,0])
    y.append(trajectory[i,1,0])
    z.append(trajectory[i,2,0])
x_truth=[]
z_truth=[]
for i in range(ground_truth.shape[0]):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11])

plt.plot(x,z, label="Proposed Method")
plt.plot(x_truth,z_truth, label="Ground Truth")
plt.title("Results")
plt.xlabel("z")
plt.ylabel("x")
plt.legend()
plt.show()


# In[14]:





# In[10]:


ground_truth.shape


# In[ ]:




