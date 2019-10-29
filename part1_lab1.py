# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:40:20 2019

@author: Usuario
"""

# Lab 1 ANNDA
# Andrés Martínez Mora, Marvin Köpff and Louis

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import time


#%%
# TASK 1

# Creation of random data

n=100 # Random points
mA=[1.0,1.0] # Mean value of class A
mB=[-1.0,-1.0] # Mean value of class B
sigmaA=0.4 # Standard deviation of class A
sigmaB=0.3 # Standard deviation of class B
classA=np.random.normal(loc=mA,scale=sigmaA,size=(n,2)).T # Normal random distribution for points of class A
classB=np.random.normal(loc=mB,scale=sigmaB,size=(n,2)).T # Normal random distribution for points of class B


# Plotting the points
plt.figure()
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Input points')


# Sequential feeding and perceptron learning rule
weights=np.random.rand((3)).T # Initial random weights
aux=np.zeros((4,2*n)) # Matrix with inputs
aux[2,:]=np.ones((1,2*n)) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classA,classB),axis=1) # Place all points from classes together
aux[3,0:n]=np.zeros((1,n)) # Label row with zeros for class A
aux[3,(n):(2*n)]=np.ones((1,n)) # Label row with ones for class A
original=np.copy(aux)
aux_T=original.T
np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label matrix
error_seq_pl=[] # Error for sequential perceptron learning rule

threshold=0.0 # Threshold for step function (activation function)
eta=0.001 # Learning step
epochs=10 # Number of iterations
print('Sequential feeding with perceptron learning rule')
start=time.time()
for j in range(epochs):
    result=[]
    for i in range(2*n):
        prod=np.dot(weights.T,inp[:,i]) # Dot product of weights and one of the inputs (sequential learning)
        if prod>threshold: # Step function
            y=1.0
        else:
            y=0.0
        result.append(y)
        # Error calculation
        weights+=eta*(labels[i]-y)*inp[:,i]
        if i==2*n-1:
            total_error=np.sum(np.abs(labels-result))/(2*n) # Total classification error
            error_seq_pl.append(total_error)
            print("Epoch: {} / Misclassification rate:{}".format(j+1,total_error))
    # Shuffle data and labels again for new epoch
    np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
    aux_shuffle=aux_T.T
    inp=aux_shuffle[0:3,:] # Input matrix (including bias)
    labels=aux_shuffle[3,:] # Label matrix
end=time.time()

print('Time for sequential and perceptron learning: {}'.format(end-start))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with sequential feeding and perceptron learning')


weight_batch=np.random.rand((3)).T # Initial random weights for batch learning

# Data shuffling for batch learning
np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label matrix
error_batch_pl=[] # Error in perceptron learning for batch mode

print('Batch feeding with perceptron learning rule')
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    error_batch_pl.append(total_error)
    print("Epoch: {} / Misclassification rate: {}".format(j+1,total_error))
    
    # Shuffle data and labels again for new epoch
    np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
    aux_shuffle=aux_T.T
    inp=aux_shuffle[0:3,:] # Input matrix (including bias)
    labels=aux_shuffle[3,:] # Label matrix

end=time.time()

print('Time for batch and perceptron learning: {}'.format(end-start))

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weight_batch[0]/weight_batch[1])*np.linspace(-2,2,400)-weight_batch[2]/weight_batch[1],color='green')
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with batch feeding and perceptron learning')


# Sequential feeding with Delta Rule

weight_delta_seq=np.random.rand((3)).T # Initial random weights for sequential learning


# Data shuffling

np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label matrix
labels[np.where(labels==0)]=-1 # Labels for Delta rule are -1 instead of 0
error_seq=[] # List keeping the misclassification rates to plot the learning curve

print('Sequential feeding with Delta rule')
start=time.time()
for j in range(epochs):
    result=[]
    for i in range(2*n):
        prod=np.dot(weight_delta_seq.T,inp[:,i]) # Dot product of weights and one of the inputs (sequential learning)
        if prod>threshold: # Step function
            result.append(1.0)
        else:
            result.append(-1.0)
        # Error calculation
        weight_delta_seq+=eta*(labels[i]-prod)*inp[:,i]
        if i==2*n-1:
            total_error=np.sum(np.abs(labels-result))/(2*n) # Total classification error
            error_seq.append(total_error) # Save error values
            print("Epoch: {} / Misclassification rate: {}".format(j+1,total_error))
        
    # Shuffle data and labels again for new epoch
    np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
    aux_shuffle=aux_T.T
    inp=aux_shuffle[0:3,:] # Input matrix (including bias)
    labels=aux_shuffle[3,:] # Label matrix
    labels[np.where(labels==0)]=-1 # Labels for Delta rule are -1 instead of 0
end=time.time()

print('Time for sequential and Delta rule: {}'.format(end-start))    

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weight_delta_seq[0]/weight_delta_seq[1])*np.linspace(-2,2,400)-weight_delta_seq[2]/weight_delta_seq[1],color='green')
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with sequential feeding and Delta rule')


weight_batch_delta=np.random.rand((3)).T # Initial random weights for batch learning

# Data shuffling for batch learning
np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label matrix
labels[np.where(labels==0)]=-1 # Labels for Delta rule are -1 instead of 0
error_batch=[] # Error for batch feeding to plot the learning curves

print('Batch feeding with Delta rule')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weight_batch_delta.T,inp) # Dot product of weights and all inputs (batch learning)
    aux=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(aux.shape)
    y_batch[np.where(aux==True)]=1.0 # Substitute True by 1
    y_batch[np.where(aux==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weight_batch_delta+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    error_batch.append(total_error)
    print("Epoch: {} / Misclassification rate: {}".format(j+1,total_error))
    
    # Shuffle data and labels again for new epoch
    np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
    aux_shuffle=aux_T.T
    inp=aux_shuffle[0:3,:] # Input matrix (including bias)
    labels=aux_shuffle[3,:] # Label matrix
    labels[np.where(labels==0)]=-1 # Labels for Delta rule are -1 instead of 0
end=time.time()

print('Time for batch and delta rule: {}'.format(end-start))

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weight_batch_delta[0]/weight_batch_delta[1])*np.linspace(-2,2,400)-weight_batch_delta[2]/weight_batch_delta[1],color='green')
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with batch feeding and Delta rule')


# Plot for learning curves in Delta rule
plt.figure()
plt.plot(range(1,epochs+1),error_seq,color='blue',label='Error in delta rule, sequential mode') # Error for sequential mode with Delta
plt.plot(range(1,epochs+1),error_batch,color='red',label='Error in delta rule, batch mode') # Error for batch mode with Delta
plt.plot(range(1,epochs+1),error_seq_pl,color='green',label='Error in perceptron learning rule, sequential mode') # Error for sequential mode with Perceptron Learning
plt.plot(range(1,epochs+1),error_batch_pl,color='black',label='Error in perceptron learning rule, batch mode') # Error for batch mode with Perceptron Learning
plt.xlabel('Epoch number')
plt.ylabel('Misclassification rate')
plt.legend()
plt.title('Learning curves for single layer perceptron')


# Perceptron without bias trained in batch mode with the Delta rule
inp=aux_shuffle[0:2,:] # Input without bias
weight_no_bias=np.random.rand((2)) # Weight vector without bias
labels[np.where(labels==0)]=-1 # Labels for Delta rule are -1 instead of 0
print('Batch feeding with Delta rule, but without biases')
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weight_no_bias.T,inp) # Dot product of weights and all inputs (batch learning)
    aux=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(aux.shape)
    y_batch[np.where(aux==True)]=1.0 # Substitute True by 1
    y_batch[np.where(aux==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weight_no_bias+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    print("Epoch: {} / Misclassification rate: {}".format(j+1,total_error))
    
end=time.time()


print('Time for batch and delta rule, but without biases: {}'.format(end-start))

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weight_no_bias[0]/weight_no_bias[1])*np.linspace(-2,2,400),color='green')
plt.scatter(classA[0,:],classA[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classB[0,:],classB[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with batch feeding and Delta rule, but without bias')


#%%

# New dataset for part 3.1.3

mC=[1.0,0.3] # Mean for class C
sigmaC=0.2 # Standard deviation for class C

mD=[0.0,-0.1] # Mean for class D
sigmaD=0.3 # Standard deviation for class D

# Initialize data for both classes
classC=np.zeros(classA.shape)
classD=np.zeros(classA.shape)

# First class. It will be made of 2 clouds of points
classC[0,0:int(np.round(n*0.5))]=np.random.normal(loc=-mC[0],scale=sigmaC,size=(1,int(np.round(n*0.5))))
classC[0,int(np.round(n*0.5))-1:-1]=np.random.normal(loc=mC[0],scale=sigmaC,size=(1,int(np.round(n*0.5))))
classC[1,:]=np.random.normal(loc=-mC[1],scale=sigmaC,size=(1,n))

left=classC[0,0:int(np.round(n*0.5))] # Left cloud of points for A
right=classC[0,int(np.round(n*0.5))-1:-1] # Right cloud of points for A

# Second class. It will be made by just one clous of points
classD=np.random.normal(loc=mD,scale=sigmaD,size=(n,2)).T



# Plotting the points
plt.figure()
plt.scatter(classC[0,:],classC[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Linearly non-separable classes')

# Construct pattern and label matrices for the given data
weights=np.random.rand((3)).T # Initial random weights
aux=np.zeros((4,2*n)) # Matrix with inputs
aux[2,:]=np.ones((1,2*n)) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classC,classD),axis=1) # Place all points from classes together
aux[3,0:n]=-np.ones((1,n)) # Label row with zeros for class A
aux[3,(n):(2*n)]=np.ones((1,n)) # Label row with ones for class A
original=np.copy(aux)
aux_T=original.T
np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label vector


# Apply delta rule in Batch Mode to whole dataset
epochs=100

print('Batch feeding with Delta rule for non-separable dataset')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    step=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(step.shape)
    y_batch[np.where(step==True)]=1.0 # Substitute True by 1
    y_batch[np.where(step==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weights+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    #error_batch.append(total_error)
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
end=time.time()

print('Time for batch and delta rule for whole dataset: {}'.format(end-start))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classC[0,:],classC[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with delta rule for non-separable dataset')

weights=np.random.rand((3)).T # Initial random weights
labels[np.where(labels)==-1]=0
print('Batch feeding with perceptron learning rule for non-separable dataset')
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
    

end=time.time()

print('Time for batch and perceptron learning: {}'.format(end-start))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classC[0,:],classC[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with perceptron learning for non-separable dataset')


# Randomly sample 75% of all the dataset
n25 = int(0.25*n)
n75 = n-n25
random_indexes25 = np.random.choice(range(0,n), n25, replace=False)

classE = np.delete(classC,random_indexes25,1) # Class C without 25% of the samples
classF = np.delete(classD,random_indexes25,1) # Class D without 25% of the samples

#classC_val = classA[:,random_indexes25]
#classD_val = classB[:,random_indexes25]


# Construct pattern and label matrices for training
aux=np.zeros((4,2*n75)) # Matrix with inputs
aux[2,:]=np.ones((1,2*n75)) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classE,classF),axis=1) # Place all points from classes together
aux[3,0:n75]=-np.ones((1,n75)) # Label row with zeros for class A
aux[3,(n75):(2*n75)]=np.ones((1,n75)) # Label row with ones for class A
original=np.copy(aux)
aux_T=original.T
np.random.shuffle(aux_T) # Maidletrix with shuffled rows for both inputs and labels
aux_shuffle=aux_T.T # Matrix with shuffled columns for both inputs and labels
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label vector


print('Batch feeding with Delta rule for non-separable dataset (-25% samples)')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    step=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(step.shape)
    y_batch[np.where(step==True)]=1.0 # Substitute True by 1
    y_batch[np.where(step==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weights+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    #error_batch.append(total_error)
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
end=time.time()

print('Time for batch and delta rule for dataset with 25% less samples: {}'.format(end-start))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classE[0,:],classE[1,:],cmap='blue',label='Class A (-25%)') # Plot for class A
plt.scatter(classF[0,:],classF[1,:],cmap='red',label='Class B (-25%)') # Plot for class B
plt.legend()
plt.title('Classification with delta rule for non-separable dataset (-25%)')


weights=np.random.rand((3)).T # Initial random weights
print('Batch feeding with perceptron learning rule for non-separable dataset (-25%)')
labels[np.where(labels)==-1]=0
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
    

end=time.time()

print('Time for batch and perceptron learning (-25%): {}'.format(end-start))

#%%

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classE[0,:],classE[1,:],cmap='blue',label='Class A (-25%)') # Plot for class A
plt.scatter(classF[0,:],classF[1,:],cmap='red',label='Class B (-25%)') # Plot for class B
plt.legend()
plt.title('Classification with perceptron learning for non-separable dataset (-25%)')


# Randomly sample 50% of just Class A
n50 = int(0.50*n)
random_indexes50 = np.random.choice(range(0,n), n50, replace=False)

classG = np.delete(classC,random_indexes50,1) # Class C without 50% of the samples
#classC_val = classC[:,random_indexes50]

#classD_train = classD


# Construct pattern and label matrices for training
aux=np.zeros((4,n+n50)) # Matrix with inputs
aux[2,:]=np.ones((1,n+n50)) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classG,classD),axis=1) # Place all points from classes together
aux[3,0:n50]=-np.ones((1,n50)) # Label row with zeros for class A
aux[3,(n50):(n+n50)]=np.ones((1,n)) # Label row with ones for class B
original=np.copy(aux)
aux_T=original.T
np.random.shuffle(aux_T) # Matrix with shuffled rows for both inputs and labels
aux_shuffle=aux_T.T # Matrix with shuffled columns for both inputs and labels
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label vector


print('Batch feeding with Delta rule for non-separable dataset (-50% samples for A)')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    step=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(step.shape)
    y_batch[np.where(step==True)]=1.0 # Substitute True by 1
    y_batch[np.where(step==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weights+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    #error_batch.append(total_error)
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
end=time.time()


good_G=np.array((np.where((labels+y_batch)==-2))).shape[1] # Number of correctly classified samples from G
good_D=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from D



accuracy_G=good_G/(classG.shape[1])
accuracy_D=good_D/(classD.shape[1])

print('Time for batch and delta rule for dataset with 50% less samples for A: {}'.format(end-start))
print('\nAccuracy for class A (-50%):{}\n'.format(accuracy_G))
print('Accuracy for class B :{}\n'.format(accuracy_D))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classG[0,:],classG[1,:],cmap='blue',label='Class A (-50%)') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with delta rule for non-separable dataset (-50% for A)')





weights=np.random.rand((3)).T # Initial random weights
print('Batch feeding with perceptron learning rule for non-separable dataset (-50% samples from A)')
labels[np.where(labels)==-1]=0
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
    

end=time.time()

print('Time for batch and perceptron learning (-50% samples from A): {}'.format(end-start))


good_G=np.array((np.where((labels+y_batch)==0))).shape[1] # Number of correctly classified samples from G
good_D=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from D

accuracy_G=good_G/(classG.shape[1])
accuracy_D=good_D/(classD.shape[1])


print('\nAccuracy for class A (-50%):{}\n'.format(accuracy_G))
print('Accuracy for class B :{}\n'.format(accuracy_D))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classG[0,:],classG[1,:],cmap='blue',label='Class A (-50%)') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with perceptron learning for non-separable dataset (-50% for A)')


# Remove 50% of samples from B
n50 = int(0.50*n)
random_indexes50 = np.random.choice(range(0,n), n50, replace=False)

classH = np.delete(classD,random_indexes50,1) # Class C without 50% of the samples

# Construct pattern and label matrices for the given data
weights=np.random.rand((3)).T # Initial random weights
aux=np.zeros((4,int(n*1.5))) # Matrix with inputs
aux[2,:]=np.ones((1,int(n*1.5))) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classC,classH),axis=1) # Place all points from classes together
aux[3,0:int(n*0.5)]=-np.ones((1,int(n*0.5))) # Label row with zeros for class A
aux[3,int(n*0.5)-1:-1]=np.ones((1,n)) # Label row with ones for class A
original=np.copy(aux)
aux_T=original.T
#np.random.shuffle(aux_T) # Matrix with shuffled columns for both inputs and labels
aux_shuffle=aux_T.T
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label vector


print('Batch feeding with Delta rule for non-separable dataset (-50% samples for B)')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    step=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(step.shape)
    y_batch[np.where(step==True)]=1.0 # Substitute True by 1
    y_batch[np.where(step==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weights+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    #error_batch.append(total_error)
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
end=time.time()


good_C=np.array((np.where((labels+y_batch)==-2))).shape[1] # Number of correctly classified samples from G
good_H=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from D



accuracy_C=good_C/(classC.shape[1])
accuracy_H=good_H/(classH.shape[1])

print('Time for batch and delta rule for dataset with 50% less samples for B: {}'.format(end-start))
print('\nAccuracy for class A :{}\n'.format(accuracy_C))
print('Accuracy for class B (-50%):{}\n'.format(accuracy_H))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classC[0,:],classC[1,:],cmap='blue',label='Class A ') # Plot for class A
plt.scatter(classH[0,:],classH[1,:],cmap='red',label='Class B (-50%)') # Plot for class B
plt.legend()
plt.title('Classification with delta rule for non-separable dataset (-50% for B)')





weights=np.random.rand((3)).T # Initial random weights
print('Batch feeding with perceptron learning rule for non-separable dataset (-50% samples from B)')
labels[np.where(labels)==-1]=0
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
    

end=time.time()

print('Time for batch and perceptron learning (-50% samples from B): {}'.format(end-start))


good_C=np.array((np.where((labels+y_batch)==0))).shape[1] # Number of correctly classified samples from G
good_H=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from D

accuracy_C=good_C/(classC.shape[1])
accuracy_H=good_H/(classH.shape[1])


print('\nAccuracy for class A :{}\n'.format(accuracy_C))
print('Accuracy for class B (-50%):{}\n'.format(accuracy_H))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classC[0,:],classC[1,:],cmap='blue',label='Class A') # Plot for class A
plt.scatter(classH[0,:],classH[1,:],cmap='red',label='Class B (-50%)') # Plot for class B
plt.legend()
plt.title('Classification with perceptron learning for non-separable dataset (-50% for B)')


# Remove 20% of samples from A in the left cloud and 80% from A in the right cloud
idxNeg = np.where(classA[1,:]<0)
idxPos = np.where(classA[1,:]>0)
lenPos= int((idxPos[0].shape)[0]*0.8)
lenNeg = int((idxNeg[0].shape)[0]*0.2)

random_indneg20 = np.random.choice(idxNeg[0], lenNeg, replace=False)
random_indpos20 = np.random.choice(idxPos[0], lenPos, replace=False)
random_ind = np.concatenate((random_indneg20, random_indpos20), axis=0)

classI = np.delete(classA,random_ind,1) # Class C without 25% of the samples


lenClassC = classI.shape[1]




# Construct pattern and label matrices for training
aux=np.zeros((4,n+lenClassC)) # Matrix with inputs
aux[2,:]=np.ones((1,n+lenClassC)) # Dedicate last row for biases (row full of ones)
aux[0:2,:]=np.concatenate((classI,classD),axis=1) # Place all points from classes together
aux[3,0:lenClassC]=-np.ones((1,lenClassC)) # Label row with zeros for class A
aux[3,(lenClassC):(n+lenClassC)]=np.ones((1,n)) # Label row with ones for class B
original=np.copy(aux)
aux_T=original.T
np.random.shuffle(aux_T) # Matrix with shuffled rows for both inputs and labels
aux_shuffle=aux_T.T # Matrix with shuffled columns for both inputs and labels
inp=aux_shuffle[0:3,:] # Input matrix (including bias)
labels=aux_shuffle[3,:] # Label vector





print('Batch feeding with Delta rule for non-separable dataset (-80% and -20% for A)')
start=time.time()  
# Classification with batch and Delta rule
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    step=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch=np.zeros(step.shape)
    y_batch[np.where(step==True)]=1.0 # Substitute True by 1
    y_batch[np.where(step==False)]=-1.0 # Substitute False by -1
    # Weight updating
    weights+=np.sum(eta*(labels-prod_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(y_batch-labels)))/(2*n) # Total classification error
    #error_batch.append(total_error)
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
end=time.time()


good_I=np.array((np.where((labels+y_batch)==-2))).shape[1] # Number of correctly classified samples from G
good_D=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from D



accuracy_I=good_I/(classI.shape[1])
accuracy_D=good_D/(classD.shape[1])

print('Time for batch and delta rule for dataset with 80% and 20% less samples for A: {}'.format(end-start))
print('\nAccuracy for class A (-80% and -20%):{}\n'.format(accuracy_I))
print('Accuracy for class B :{}\n'.format(accuracy_D))


# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classI[0,:],classI[1,:],cmap='blue',label='Class A (-80% and -20%)') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with delta rule for non-separable dataset (-80% and -20% for A)')





weights=np.random.rand((3)).T # Initial random weights
print('Batch feeding with perceptron learning rule for non-separable dataset (-80% and -20% for A)')
labels[np.where(labels)==-1]=0
start=time.time()  
# Classification with batch and perceptron learning
for j in range(epochs):
    prod_batch=np.dot(weights.T,inp) # Dot product of weights and all inputs (batch learning)
    y_batch=np.greater(prod_batch,threshold) # Array with outputs after applying the step function
    y_batch[np.where(y_batch==True)]=1.0 # Substitute True by 1
    y_batch[np.where(y_batch==False)]=0.0 # Substitute False by 0
    # Weight updating
    weight_batch+=np.sum(eta*(labels-y_batch).T*inp,1)
    # Error computation
    total_error=(np.sum(np.abs(labels-y_batch)))/(2*n) # Total classification error
    print("Epoch: {} / Accuracy rate: {}".format(j+1,1-total_error))
    

end=time.time()

print('Time for batch and perceptron learning (-80% and -20% for A): {}'.format(end-start))


good_I=np.array((np.where((labels+y_batch)==2))).shape[1] # Number of correctly classified samples from G
good_D=np.array((np.where((labels+y_batch)==0))).shape[1] # Number of correctly classified samples from D

accuracy_I=good_I/(classI.shape[1])
accuracy_D=good_D/(classD.shape[1])


print('\nAccuracy for class A (-80% and -20% for A):{}\n'.format(accuracy_I))
print('Accuracy for class B :{}\n'.format(accuracy_D))

# Visualization plot
plt.figure()
plt.plot(np.linspace(-2,2,400),-(weights[0]/weights[1])*np.linspace(-2,2,400)-weights[2]/weights[1],color='green')
plt.scatter(classI[0,:],classI[1,:],cmap='blue',label='Class A (-80% and -20%)') # Plot for class A
plt.scatter(classD[0,:],classD[1,:],cmap='red',label='Class B') # Plot for class B
plt.legend()
plt.title('Classification with perceptron learning for non-separable dataset (-80% and -20% for A)')