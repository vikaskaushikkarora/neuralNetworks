# Neural Network To Recognise The Handwritten Digits
# Only 1 Hidden Layer Neuron 

#Import Libraries
import numpy as np
import idx2numpy
from numpy import sin , cos , exp , pi 
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt


#Define parameters
x=784
h=100
y=10


#Import learned weights and biases
wb=np.loadtxt('weightsAndBiases.txt')

a=0
Hb=wb[a:a+h]
Hb.shape=(1,h)
a=a+h
Yb=wb[a:a+y]
Yb.shape=(1,y)
a=a+y
XH=wb[a:a+x*h]
XH.shape=(x,h)
a=a+x*h
HY=wb[a:a+h*y]
HY.shape=(h,y) 


# Defining Sigmoid Function
def sig(z):
	y=(1+exp(-z))**(-1)
	return y
	
#Defining Derivative of Sigmoid
def dsig(z):
	w=exp(-z)*(1+exp(-z))**(-2)
	return w

#Defining Output Function
def output(X,Hb,Yb,XH,HY):
	Hz=np.dot(X,XH)+Hb
	H=sig(Hz)
	Yz=np.dot(H,HY)+Yb
	Y=sig(Yz)
	return Y

#Prepare Test Data
file = 'train-images.idx3-ubyte'
Data= idx2numpy.convert_from_file(file)
Data=Data*(256)**(-1)

file1 = 'train-labels.idx1-ubyte'
b= idx2numpy.convert_from_file(file1)

X1=Data[59997,:,:] #You can take any image from 59995 to 59999, I have selected 5 images for testing.
X=X1.copy()
X.shape=(1,784)
Y=output(X,Hb,Yb,XH,HY)
a=max(Y[0,0],Y[0,1],Y[0,2],Y[0,3],Y[0,4],Y[0,5],Y[0,6],Y[0,7],Y[0,8],Y[0,9])

A=-100
for j in range(10):
	if Y[0,j] == a :
		A=j


print("The Predicted Number is : " , A)

plt.imshow(X1, interpolation='nearest')
plt.show()  #Show/plot that image
