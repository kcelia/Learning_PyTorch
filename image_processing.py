import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#rtype : Image.open("img_1.jpeg") 
#PIL.JpegImagePlugin.JpegImageFile
#im = np.array(Image.open("img_1.jpeg"), dtype=float)
#image bizarre
#we can display the PIL object without np.array
im = np.array(Image.open("img_1.jpeg"))
#help(np.array)

#convert un image to grayscale
Image.open("img_1.jpeg").convert('L').save("n.jpg")
im2 = Image.open("n.jpg")
plt.imshow(im2) ; plt.show()

#creat thumbnails

#(left, upper, right, lower)
box = (100, 100, 400, 400)
im3 = im2.crop(box)
plt.imshow(im2) ; plt.show()
plt.imshow(im3) ; plt.show()

#hist of distribution of pixel values
plt.hist(np.array(im2).flatten(), bins=128)
plt.show()