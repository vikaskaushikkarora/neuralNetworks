# Neural Network (1-1-1) #
# Cross Entropy Cost Function
# Regularization Technique used

# Check if the Number is greater than zero OR not 

#______________________________

import numpy as np
from numpy import exp , sin , cos , tan , pi , log

print ('## Neural Network Training Results at the end of each Training Batch ##')

#______________________________

# initial value of weights and biases which will be improved
w1=10
b1=5
w2=7
b2=5
P=2*10**(-6) # Regularization constant

# Defining Sigmoid Function
def sig(z):
	y=(1+exp(-z))**(-1)
	return y
	
def dsig(z):
	y=(exp(-z))*(1+exp(-z))**(-2)
	return y

#______________________________

# Function which improves the values of weight and bias by Gradient Descent and Backpropagation Method 

def improve(a,t,w1,b1,w2,b2):
	z1=w1*a+b1
	A1=sig(z1)
	z2=w2*A1+b2
	A=sig(z2)

	gb2=(A-t) # gradient of cost function in direction of w1
	gw2=gb2*A1 + P*w2  # gradient of cost function in direction of b1
	gb1=gb2*w2*dsig(z1)
	gw1=gb2*w2*dsig(z1)*a + P*w1
	return gw1,gb1,gw2,gb2

#______________________________

# Training Data
m=500000 #Number of Training Data Points
X=np.zeros(m) #Matrix Containing Test Numbers
Y=np.zeros(m) #Matrix Containing Truth Values 

for i in range(m):
	a=(-1)**np.random.randint(1,1000)
	if a > 0 :
		X[i]=((-1)**np.random.randint(1,1000))*np.random.rand()
	else :
		X[i]=10*((-1)**np.random.randint(1,1000))*np.random.rand()
	if X[i] > 0 :
		Y[i]=1
	else :
		Y[i]=0

#______________________________

#Stochastic Gradient Descent Method	
n=len(X)
N=1000 # Number of Batches
for j in range(N):
	gw1_0=0
	gb1_0=0
	gw2_0=0
	gb2_0=0
	cost0=0
	for i in range(j*int(n*N**(-1)),(j+1)*int(n*N**(-1))):
		#Applying Gradient Descent for each test data anf then taking average of gradient w and gradient b
		a=X[i]
		t=Y[i]
		alpha,beta,gamma,delta=improve(a,t,w1,b1,w2,b2)
		gw1_0=gw1_0+alpha
		gb1_0=gb1_0+beta
		gw2_0=gw2_0+gamma
		gb2_0=gb2_0+delta
		z1=w1*a+b1
		A1=sig(z1)
		z2=w2*A1+b2
		A=sig(z2)
		cost0=cost0-(t*log(1-A)+(1-t)*log(1-A)) + 0.5*P*(w1**2+w2**2)
	gw1=gw1_0*(int(n*N**(-1)))**(-1)
	gb1=gb1_0*(int(n*N**(-1)))**(-1)
	gw2=gw2_0*(int(n*N**(-1)))**(-1)
	gb2=gb2_0*(int(n*N**(-1)))**(-1)
	cost=cost0*(int(n*N**(-1)))**(-1)
	w1=w1-1.5*gw1
	b1=b1-1.5*gb1
	w2=w2-1.5*gw2
	b2=b2-1.5*gb2
	#print ('\nw1 is : ' , w1)
	#print ('b1 is : ' , b1)
	#print ('w2 is : ' , w2)
	#print ('b2 is : ' , b2)
	print ('cost is : ' , cost)
	
#______________________________

p=3
while p > 2 :
	x=float(input('\nEnter the number you want to check : '))
	z1=w1*x+b1
	A1=sig(z1)
	z2=w2*A1+b2
	A=sig(z2)
	print ('o\nValue of Activation for the output Neuron is : ',A)
	if A < 0.1 :
		print('\nThe number is less than zero !')
	elif A> 0.9 :
		print('\nThe number is greater than zero !')
	else :
		print ('\nYem kyam ho rhan haim : Unable to decide !')
