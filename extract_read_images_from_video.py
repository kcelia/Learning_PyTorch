import cv2 #print(cv2.__version__)
import os



"""
Main Functions 
video = cv2.VideoCapture(File_path)
success, image = vidcap.read()
imwrite(filename, img[, params]) 

"""
FILE_VID = "test/video1.avi"
FILE_DIR = "./first_test_extract_images_from_video"
AVI = '.avi'
JPEG = '.jpg'

#VideoCapture(File_path) : Read the video(.XXX format)
#rtype: tuple(boolean, numpy.ndarray)


vidcap = cv2.VideoCapture(FILE_VID)
"""
In [16]: vidcap.read()[1].shape
Out[16]: (240, 320, 3)
"""
success, image = vidcap.read()
"""
In [25]: image.shape
Out[25]: (240, 320, 3)
"""

def createFolder(file_name):
    try:
        if not os.path.exists(file_name):
            os.makedirs(file_name)
       
    except OSError: 
        print ('Error: Creating directory of data') 

count = 0
success = True
tmp = 0
STRIDE  = 10
createFolder(FILE_DIR)
while success:
    if count % STRIDE == 0:
        tmp +=1
        file_im = "{}/{}_frame_{}{}".format(FILE_DIR, FILE_VID[:-len(AVI)], tmp, JPEG)
        cv2.imwrite(file_im, image)  
    success, image = vidcap.read()
    count += 1

vidcap.release() 




#Rappel

#function_object = lambda arguments : expression
add = lambda x, y : x + y
add(1, 2) # out: 3
#map(function_object, iterable1, iterable2,...)
list(map(add, [1, 2, 3, 4], [1, 2, 3, 4]))
#filter(function_object, iterable)


import random
l1 = [1, 5, 8]
l2 = [0, 0, 1]

l3 = list(zip(l1, l2))
random.shuffle(l3)
trainx, trainy = list(zip(*l3))


# /t/ with [p], [k], [f], [th], [gh], [s], [sh], [ch]
# /ID/ with [t], [d]
# /d/ with other

