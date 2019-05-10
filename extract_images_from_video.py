import os
import cv2  
import time
import copy 
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from glob import glob
from tqdm import tqdm




path_original_video="UCF"
ext_video = '.vid'

def reset(path="./image_extracted_from_video"):
    if os.path.exists(path):
        print("full directory")
        os.popen('rm -r ./' + path )
    else:
        print("empty directory")

def createFolder(file_name):
    try:
        if not os.path.exists(file_name):
            os.makedirs(file_name)
       
    except OSError: 
        print ('Error: Creating directory of data') 

def extarct_images_from_video(video_name, video_num, path, \
    img_size=(224, 224), ext='.jpeg', stride=25):
    vidcap = cv2.VideoCapture(video_name)
    count = 0
    success, image = vidcap.read()
    createFolder("{}/frame_{}".format(
                path, video_num))
    while success:
        if count % stride == 0:
            file_im = "{}/frame_{}/img_{}{}".format(
                path, video_num, 
                count // stride + 1,
                ext)

            image = cv2.resize(image, img_size)
            cv2.imwrite(file_im, image)  
        success, image = vidcap.read()
        count += 1
    vidcap.release() 

def exaction_image(path="./image_extracted_from_video/", 
    path_original_video="UCF", 
    path_extr_img="images_extracted"):

    actions = list(map(lambda x : x.split("\\")[-1], glob(path_original_video + "/*")))
    #actions2 = glob(path_original_video + "/*")

    for action in actions:
        videos = list(map(lambda x : x.split("\\")[-1], 
        glob(path_original_video + '/' + action + "/*.avi")))
      
        for video_num, video in enumerate(glob(path_original_video + '/' + action + "/*.avi")):
            path_extr_img = path + action+"/"+video[0][4:-4]
            extarct_images_from_video(video, video_num, 
            path_extr_img)




reset(path="./image_extracted_from_video")
exaction_image()

#***************************************************************************#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#normalize images 
img_file = "./image_extracted_from_video/ApplyEyeMakeup/frame_0/img_1.jpeg"
img = plt.imread(img_file)
plt.imshow(img) ; plt.show()

"""
In [78]: plt.imread(img_file).shape
Out[78]: (256, 340, 3) 
"""
#Composes several transforms together.
"""
Parameters: transforms (list of Transform objects) 
list of transforms to compose.

"""
img = plt.imread(img_file)

#normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])

pret = transforms.Compose([
    transforms.ToPILImage(),
    #input : PILE
    transforms.CenterCrop(224),
    #input : PILE <!> Resize transform returns an image
    #<!> resizing an image changes the spatial ratio
    # Cropping at the image center is better
    #transforms.Resize(size=(100,100), interpolation=2),
    transforms.ToTensor(),
    #The ToTensor transform should come before the Normalize transform, 
    # since the latter expects a tensor
    # <!> Normalize transform returns an image
    #normalize or
    transforms.Normalize(
        mean=[.485, .456, .406],
        std=[.229, .224, .225]
    )
])

img_prep = pret(img).detach().permute(1, 2, 0).numpy()
plt.imshow(img_prep) ; plt.show()

"""
For the original VGG16 network, it output 1000 activations.
We could use these 1000 activations as a feature representation 
to train and test the RNN model that will be introduced later. 
However, these activations are too object specific, 
and may not generalize well to new classification tasks. 
Therefore, the 4096 activations from the last 2nd fully-connected layer 
are a better choice. 
"""
#VGG16 takes images with size of (224, 224) as inputs and ocurrently they are (256, 340)
vgg16 = models.vgg16(pretrained=True)

"""
In [140]: vgg16.classifier
Out[140]: 
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace)
  (2): Dropout(p=0.5)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)

"""
#Modify the classifier of the network
"""
REMOVE
  (2): Dropout(p=0.5)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)

"""
feature_map = list(vgg16.classifier.children())
feature_map = feature_map[:-3] #feature_map.drop()
vgg16.classifier = nn.Sequential(*feature_map)
vgg16 = vgg16.to(device)




def extarct_features_from_images(
    model, batch_size=2,
    dir= "./image_extracted_from_video/",
    path_img="./image_extracted_from_video/*/frame_?", 
    path_features=["./features_images2/"]):
    
    frames = glob(path_img)
    images2 = list(map(lambda x : x.split("\\"), glob(path_img)))
    path_out = list(map(lambda x : "/".join(path_features + x[1:-1]), images2))
    
    
    for i, frame in enumerate(frames):
        batch = []
        images = glob(frame + "/img_*.jpeg")
        for image in images:
            img = cv2.imread(image)
            batch.append(pret(img).detach().numpy())

        image_batch =  np.reshape(batch, (len(batch), 3, 224, 224))        
        image_batch = torch.from_numpy(image_batch)
        print("here 1")
        features = model(image_batch)
        createFolder(path_out[i])
        print("here 2")
        io.savemat(path_out[i] + "/frame_" + str(i), {"Features":features})
        
        break
extarct_features_from_images(vgg16)
        

import zipfile

zipname ="testzip.zip"

zipobj = zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED)
t = glob("./features_images3/Archery/*")[5]
io.loadmat(t)['Features']

#Creation du batch 

"""
LSTM [25] is designed explicitly to avoid the longterm dependency problem, 
and remembering information for long periods of time is practically its default behavior.

A common LSTM unit has a cell, 
1. an input gate
2. an output gate
3. a forget gate.

The cell remembers values over arbitrary time intervals 
and the three gates regulate the flow of information into and out of the cell. 
LSTM is well-suited to classifying, processing and making predictions based on time series data, 
since there can be lags of unknown duration between important events in a time series. 

copy.deepcopy(model.state_dict()).


"""