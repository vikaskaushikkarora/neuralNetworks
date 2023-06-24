# Neural Network To Recognise The Handwritten Digits
# Only 1 Hidden Layer Neuron 
# Training 

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin , cos , exp , pi , log
from numpy import random as rd
import idx2numpy

#Define parameters
x=784 #Input layer neurons
h=100 #Hidden layer neurons 
y=10 #output layer neurons 
P=1.5*10**(-3) #Regularization Constant 

#Initial Weights and Biases :- 
#(Selected Randomly)
Hb=rd.randint(-5,5,size=(1,h))
Yb=rd.randint(-5,5,size=(1,y))
HY=rd.randint(-5,5,size=(h,y))

#Gaussianly Distributed Weights between Input and Hidden Layer
W=np.loadtxt('randomNumbers.txt')
XH=np.zeros((x,h))
for i in range(x):
	for j in range(h):
		XH[i,j]=W[h*i+j]
		
# Defining Sigmoid Function
def sig(z):
	y=(1+exp(-z))**(-1)
	return y
	
#Defining Derivative of Sigmoid
def dsig(z):
	w=exp(-z)*(1+exp(-z))**(-2)
	return w

#Defining Cost Function (Cross Entropy Cost Function)
def cost(X,A,Hb,Yb,XH,HY):
	Hz=np.dot(X,XH)+Hb
	H=sig(Hz)
	Yz=np.dot(H,HY)+Yb
	Y=sig(Yz)
	return -1*(np.sum(A*np.log(Y)+(1-A)*np.log(1-Y))) + 0.5*P*np.sum(XH**2) + 0.5*P*np.sum(HY**2)

#Defining Improve Function
def improve(X,A,Hb,Yb,XH,HY):
	Hz=np.dot(X,XH)+Hb
	H=sig(Hz)
	Yz=np.dot(H,HY)+Yb
	Y=sig(Yz)

	XT=X.copy()
	XT.shape=(x,1)
	HT=H.copy()
	HT.shape=(h,1)
	HYT=HY.copy()
	HYT.shape=(y,h)
	
	gYb=(Y-A)
	gYb.shape=(1,y)
	gHY=np.dot(HT,gYb)+P*HY
	
	gHb=(np.dot(gYb,HYT))*dsig(Hz)
	gXH=np.dot(XT,gHb)+P*XH

	return gHb,gYb,gXH,gHY

#Import training data
file = 'train-images.idx3-ubyte'
Data= idx2numpy.convert_from_file(file)
Data=Data*(256)**(-1)

file1 = 'train-labels.idx1-ubyte'
b= idx2numpy.convert_from_file(file1)

B=np.zeros((60000,10))
for i in range(60000):
	B[i,b[i]] = 1

#Stochastic Gradient Descent Method
n=59000
N=int(n*100**(-1))
costmatrix=np.zeros(N)
abcmatrix=np.arange(0,N,1)

for j in range(N):
	gHb_0=np.zeros(h)
	gYb_0=np.zeros(y)
	gXH_0=np.zeros((x,h))
	gHY_0=np.zeros((h,y))
	cost0=0
	
	for i in range(j*int(n*N**(-1)),(j+1)*int(n*N**(-1))):
		X=Data[i,:,:]
		X.shape=(1,784)
		A=B[i,:]
		
		#Applying Gradient Descent for each test data and then taking average ( over the whole batch ) of gradient w and gradient b
		alpha,beta,gamma,delta=improve(X,A,Hb,Yb,XH,HY)
		
		gHb_0=gHb_0+alpha
		gYb_0=gYb_0+beta
		gXH_0=gXH_0+gamma
		gHY_0=gHY_0+delta
		cost0=cost0+cost(X,A,Hb,Yb,XH,HY)
	
	gHb=gHb_0*(int(n*N**(-1)))**(-1)
	gYb=gYb_0*(int(n*N**(-1)))**(-1)
	gXH=gXH_0*(int(n*N**(-1)))**(-1)
	gHY=gHY_0*(int(n*N**(-1)))**(-1)
	C=cost0*(int(n*N**(-1)))**(-1)
	
	h0=1 #step-size
	Hb=Hb-h0*gHb
	Yb=Yb-h0*gYb
	XH=XH-h0*gXH
	HY=HY-h0*gHY
	
	costmatrix[j]=C
	
	print (j+1 ,'/', N, ' cost is ' , C)
	



#Update weights and biases after the training
file=open('weightsAndBiases.txt','w')

for i in range(h):
	file.write(str(Hb[0,i])+"\n")

for i in range(y):
	file.write(str(Yb[0,i])+"\n")

for i in range(x):
	for j in range(h):
		file.write(str(XH[i,j])+"\n")

for i in range(h):
	for j in range(y):
		file.write(str(HY[i,j])+"\n")

file.close()

plt.plot(abcmatrix,costmatrix,'-r')
plt.xlabel("Number of Batchs Trained For")
plt.ylabel("Cost")
plt.title("Cost with Training")
plt.show()
