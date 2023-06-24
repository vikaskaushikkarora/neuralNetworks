# Neural Network To Tell If The Number Is Greater Than Zero Or Not

We have five different Neural Networks here :

### 0h : a very basic neural network with zero hidden layers
### 0hc : a neural network with zero hidden layers and cross entropy cost function
### 1h : a very basic neural network with one hidden layer
### 1hc : a neural network with one hidden layer and cross entropy cost function 
### 1hcr : a neural network with one hidden layer , cross entropy cost function and regularization technique 

( Here, there is just a single neuron for the hidden layer! )

As you can see that the zero hidden layer neural network with cross entropy cost function is best for this problem because adding more layers overfits the situation and to solve that , we have to we have to apply regularization techniques which is a bulky business for this little classification problem ( However it is doable and it will also give good results with more training ) . We have specifically used the cost function as cross entropy cost function to make the learning process faster , the results of which can be seen clearly . I almost got no better results with the weight initiallization techniques when I took multi-hidden layer neural networks (which is strictly not necessary for this little problem) which are not given in this folder !
