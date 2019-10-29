# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:21:39 2019

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CODE FOR MULTI-LAYER PERCEPTRON

# Generation of patterns and targets

nx=20 # Number of points in X
ny=20 # Number of points in Y
ndata=nx*ny # Total number of data
x=np.linspace(-5,5,num=nx) # X points
y=np.linspace(-5,5,num=ny) # Y points
z=np.exp(-np.outer(x,x)*0.1)*(np.exp(-np.outer(y,y)*0.1).T)-0.5 # Target values
targets=np.reshape(z,(1,ndata)) # Target vector



xx,yy=np.meshgrid(x,y)
zz=np.exp(-(xx**2)*0.1)*(np.exp(-(yy**2)*0.1).T)-0.5

ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,zz)


patterns=np.zeros((3,ndata))
patterns[0,:]=np.reshape(xx,(1,ndata))
patterns[1,:]=np.reshape(yy,(1,ndata))
patterns[2,:]=np.ones((1,ndata)) # Include bias

def activation(vector, derivative=False):
    # Compute activation function or its derivative
    out=(2./(1+np.exp(-vector)))-1
    if derivative==False:
        return out
    else:
        return (np.multiply((1+out),(1-out)))/2


# Building the network
nodes1=25 # Number of nodes for first layer
nodes2=1 # Number of nodes for second layer
weights_layer1=2*np.random.rand(nodes1,3).T-1 # Weights for first layer
weights_layer2=2*np.random.rand(nodes2,nodes1).T-1 # Weights for second layer

# Parameters for learning
alpha=0.9 # Momentum coefficient
epochs=100 # Number of epochs
eta=0.00001 # Learning rate
error_list=[]
for j in range(epochs):
    # Forward pass
    out_layer1=activation(np.dot(patterns.T,weights_layer1),derivative=False)
    out=activation(np.dot(out_layer1,weights_layer2),derivative=False)
    error=0.5*np.sum((targets-out)**2)/targets.shape[1] # Mean square error
    error_list.append(error)
    if np.remainder(j,10)==0:
        # Error printing
        print('Epoch: {} / Error: {}\n'.format(j,error))
    # Backward pass
    delta_layer2=np.multiply((out.T-targets),activation(out,derivative=True).T)
    delta_layer1=np.multiply((np.outer(weights_layer2.T,delta_layer2)).T,activation(out_layer1,derivative=True))
    
    # Weight update
    if j==0:
        d_weights2=-eta*np.dot(out_layer1.T,delta_layer2.T) # Variation for weights of second layer
        #print(patterns.shape,delta_layer1.shape)
        d_weights1=-eta*np.dot(patterns,delta_layer1) # Variation for weights of first layer
        weights_layer1+=d_weights1
        weights_layer2+=d_weights2
    else: # Incorporate momentum on the algorithm
        d_weights2=d_weights2*alpha - (1-alpha)*np.dot(out_layer1.T,delta_layer2.T) # Variation for weights of second layer
        d_weights1=d_weights1*alpha - np.dot(patterns,delta_layer1) # Variation for weights of first layer
        weights_layer1+=eta*d_weights1
        weights_layer2+=eta*d_weights2



# Learning curve plot

plt.figure()
plt.plot(range(epochs),error_list)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Learning curve')

print(xx.shape)

# Surface plot
plt.figure()
net_out=np.reshape(out,(nx,ny))
ax.plot_surface(xx,yy,net_out)