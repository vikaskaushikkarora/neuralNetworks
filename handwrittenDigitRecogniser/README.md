# Hand Written Digit Recognising Neural Network

In this Neural Network, we have 784 input neurons (for each pixel of 28x28 image), 100 hidden layer neurons and 10 output neurons coressponding to each digit (0-9) such that if this neural network recognises the image as an 8, the 8th output neuron is lit up the most and the others have very low values. 

We use **Backpropagation Method** and **Stochastic Gradient Descent Method** to train our model. The cost function is **Cross-Entropy Cost Function**, of course due to faster learning characteristics in comparision to normal square cost function. We give random weights and biases initially to start with our training, except for the weights and biases connecting input neurons to the hidden neurons, where we give gaussian distributed weights, which is helpful in training.

We also use **Regularization Technique** to overcome the overfitting situation. Among other things, we have labelled training data of **60,000** images from **MNIST Data Set**. I selected 1000 images for testing purpose and remaining 59000 images are used for training. Some extra library of python like idx2numpy are used in order to extract the image pixels as a matrix such that we can apply mathematical operations on that.

We export the weights and biases into a txt file which is really the outcome of the training of the Neural Network. This is what our Neural Network has learnt. And while we need to test our Neural Network, all we need are these weights and biases.

I did an accuracy testing on this Neural Network. I gave this Neural Network a total of about 1000 images and it predicted the correct results almost **95 percent** times, which is a very good thing!

## How to run the programme ?

#### Create random numbers for some weights and biases initially (createGaussianDistributedRandomWeights.py)

#### Train the neural network on the given labelled data (nnTraining.py)

#### Test the neural network for a single image (nnTesting.py)

#### Accuracy test is performed on 995 images (nnAccuracyTesting.py)
