import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng

image=cv2.imread('C:\\Users\\omare\\EPLTeacking\\output_images\\cropped_image8.jpg')
image=image[int(image.shape[0]/4):int(image.shape[0]/2),:]



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

edged = cv2.Canny(gray, 30,200) 

# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    

print("Number of Contours found = " + str(len(contours))) 

count=0
val_contours=[]
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    center_x,center_y=image.shape[0]/2,image.shape[1]/2
    diff_x=center_x-(x+w)
    diff_y=center_y-(y+h)
    print('difference in x for contour number ' + str(count) + ' is ' + str(diff_x))
    print('difference in y for contour number ' + str(count) + ' is ' + str(diff_y))

    dist=np.sqrt((diff_x**2)+(diff_y**2))

    print('The distance between the points are' + str(count) + ' ' + str(dist))

    if dist<15:
        val_contours.append(cnt)
    count+=1

images=[]
for cnt in val_contours:
    x,y,w,h=cv2.boundingRect(cnt)
    img=image[x:x+w,y:y+h]
    images.append(img)


plt.imshow(images[1])

plt.show()