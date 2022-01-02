# Image Show

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# create an array A from the image 
A = mpimg.imread('sample.jpg')
print(A)

#Plot this array as image in Matplotlib
imgplot = plt.imshow(A)
plt.show()
