# We need to create gaussianly distributed random weights between input and first hidden layer neurons to reduce the saturation 
# That is why we are creating random numbers between -1 and 1 following a gaussian distribution of the given width !

import numpy as np 
from numpy import exp , log , pi , sin , cos

x=np.linspace(-1,1,100)
m=len(x)

#Gaussian Distribution 
def f(x):
	z=exp(-N0*x**2)
	return z
	
X=784
H=100
Y=10

N0=X*H
N=X*H+2000
h=x[2]-x[1]
print(N0)

def random(n,x,z):
	# A=Area under the curve of distribution function 
	A=0
	for j in range(m):
		A=A+z[j]*h
	# Break up the x axis into small intervals for which total number of random numbers within is constant ......
	frac=np.zeros(m)
	# a is an array which contains number of random numbers per division in x axis
	a=np.zeros(m,dtype='int')
	
	for i in range(m):
		frac[i]=(z[i]*h)*A**(-1)
		a[i]=int(n*frac[i])
	b=sum(a)
	print(b)
	
	# array containing random numbers is rd
	rd=np.zeros(b)
	l=0
	for j in range(m):
		for i in range (a[j]):
			rd[l+i]=x[j]+h*np.random.rand()
		l=l+a[j]
	return rd
	
rd=random(N,x,f(x))
print(rd)

file=open('randomNumbers.txt','w')
for i in range(N0):
	file.write(str(rd[i])+"\n")
file.close()
