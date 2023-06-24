# Neural Network (1-nill-1) #

# Check if the Number is greater than zero OR not 

#______________________________

import numpy as np
from numpy import exp , sin , cos , tan , pi
import matplotlib.pyplot as plt

print ('## Neural Network Training Results at the end of each Training Batch ##')

#______________________________

w = 1 # initial value of weight which will be improved
b = 0.1 #initial value of bias which will be improved

# Defining Sigmoid Function
def sigmoid(z):
	y=(1+exp(-z))**(-1)
	return y

#______________________________

# Function which improves the values of weight and bias by Gradient Descent and Backpropagation Method 

def improve(a,t,w,b):
	z=w*a+b
	A=sigmoid(z)
	x=2*(A-t)*exp(-z)*(1+exp(-z))**(-2)
	gw=x*a  # gradient of cost function in direction of w
	gb=x  # gradient of cost function in direction of b
	return gw,gb

#______________________________

# Training Data
m=100000 #Number of Training Data Points
X=np.zeros(m) #Matrix Containing Test Numbers
Y=np.zeros(m) #Matrix Containing Truth Values 

for i in range(m):
	X[i]=((-1)**np.random.randint(1,1000))*np.random.rand()
	if X[i] > 0 :
		Y[i]=1
	else :
		Y[i]=0

#______________________________

#Stochastic Gradient Descent Method
	
n=len(X)
N=1000 # Number of Batches
for j in range(N):
	gw0=0
	gb0=0
	cost0=0
	for i in range(j*int(n*N**(-1)),(j+1)*int(n*N**(-1))):
		#Applying Gradient Descent for each test data and then taking average of gradient w and gradient b
		a=X[i]
		t=Y[i]
		alpha,beta=improve(a,t,w,b)
		gw0=gw0+alpha
		gb0=gb0+beta
		cost0=cost0+(sigmoid(a*w+b)-t)**2
	gw=gw0*(int(n*N**(-1)))**(-1)
	gb=gb0*(int(n*N**(-1)))**(-1)
	cost=cost0*(int(n*N**(-1)))**(-1)
	w=w-50*gw
	b=b-50*gb
	print ('\nw is : ' , w)
	print ('b is : ' , b)
	print ('cost is : ' , cost)
	
		
#______________________________

p=3
Input=np.linspace(-0.1,0.1,2000)
z=w*Input+b
A=sigmoid(z)
plt.plot(Input,A)
plt.savefig('/sdcard/1_work/plot.png',dpi=300)

while p > 2 :
	x=float(input('\nEnter the number you want to check : '))
	z=w*x+b
	A=sigmoid(z)
	print ('\nValue of Activation for the output Neuron is : ',A)
	if A < 0.1 :
		print('\nThe number is less than zero !')
	elif A> 0.9 :
		print('\nThe number is greater than zero !')
	else :
		print ('\nYem kyam ho rhan haim : Unable to decide !')
