# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:13:32 2019

@author: Usuario
"""

# Lab 3 ANNDA: Hopfield Networks. Parts 3.5 & 3.6

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Necessary functions:

def weight_matrix(vectors):
    """Compute the weight matrix for an Hopfield Network
    
    Arguments:
        vectors {`np.array`} -- an array of vectors in a row format,
                                for ex np.array([[1,2,3],[4,5,6]]),
                                for 2 vectors of 3 elements
    
    Returns:
        `np.array` -- A square symmetric matrix of size NxN
                      where N is the length of 1 vector
    """
    size = vectors.shape[-1]
    w = np.zeros(shape=(size,size))
    for v in vectors:
        w += np.outer(v,v)
    w[np.diag_indices(size)] = 0
    return w/size

def activity_weight_matrix(vectors,activity):
    """Compute the weight matrix for an Hopfield Network
    
    Arguments:
        vectors {`np.array`} -- an array of vectors in a row format,
                                for ex np.array([[1,2,3],[4,5,6]]),
                                for 2 vectors of 3 elements
    
    Returns:
        `np.array` -- A square symmetric matrix of size NxN
                      where N is the length of 1 vector
    """
    size = vectors.shape[-1]
    w = np.zeros(shape=(size,size))
    for v in vectors:
        w += np.outer(v-activity,v-activity)
    w[np.diag_indices(size)] = 0
    return w

# this function is used to compute the sign function in a effective numpy way
sgn = np.vectorize(lambda x: -1 if x < 0 else 1)


def pretty_print(pict):
    """Pretty print the images stored in pict.dat
    
    Arguments:
        pict {`np.array`} -- an binary array of size 1024
    """
    img = np.reshape(pict, newshape=(32,32))
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.show()




class Hopfield(object):
    """A Object represnting a Hopfield network
    """
    def __init__(self, patterns,W):
        """Constructor
        
        Arguments:
            patterns {`np.array`} -- an array of vectors in a row format, which
                                     are the patterns to store in the network
                                     for ex np.array([[1,2,3],[4,5,6]]),
                                     for 2 vectors of 3 elements
        """
        self.patterns = patterns
        self.W = weight_matrix(patterns) 

    def sync_recall(self, patterns, max_iter=100):
        """Performs synchroneous recall. Works with muliple patterns at once
        
        Arguments:
            patterns {`np.array`} -- an array of vectors in a row format, which
                                are the patterns to retrieve with the network
        
        Keyword Arguments:
            max_iter {int} -- Max number iteration to stop if the network doesnt converge (default: {100})
        
        Returns:
            `np.array` --  an array of vectors in a row format, which
                                are the patterns retrieved by the network
        """
        for _ in range(max_iter):
            n_patterns = sgn(np.dot(patterns, self.W))
            if np.array_equal(n_patterns, patterns):
                 return n_patterns
            patterns = n_patterns
        return n_patterns
    
    def sync_recall_bias(self, patterns, bias, max_iter=100):
        """Performs synchroneous recall. Works with muliple patterns at once
        
        Arguments:
            patterns {`np.array`} -- an array of vectors in a row format, which
                                are the patterns to retrieve with the network
        
        Keyword Arguments:
            max_iter {int} -- Max number iteration to stop if the network doesnt converge (default: {100})
        
        Returns:
            `np.array` --  an array of vectors in a row format, which
                                are the patterns retrieved by the network
        """
        for _ in range(max_iter):
            n_patterns = 0.5+0.5*np.sign(np.dot(patterns, self.W)-bias)
            if np.array_equal(n_patterns, patterns):
                 return n_patterns
            patterns = n_patterns
        return n_patterns

    def async_recall(self, pattern, max_iter=1000, verbose=False):
        """Performs asynchronous recall. Works with only one pattern at once.
        
        Arguments:
            pattern {`np.array`} -- a pattern to recall, stored in a row format
        
        Keyword Arguments:
            max_iter {int} -- Max iterations to perform (default: {1000})
            verbose {bool} -- If True, this will print an image of the pattern every 500 iteration (default: {False})
        
        Returns:
            `np.array` -- A vector in a row format, the retrieved pattern
        """
        n_pattern = np.array(pattern, copy=True)
        energy = []
        for i in range(max_iter):
            firing_neuron = np.random.randint(0,pattern.shape[-1])
            new_n_val = sgn(np.dot(pattern, self.W[firing_neuron]))
            n_pattern[firing_neuron] = new_n_val
            if verbose and i%500==0:
                pretty_print(n_pattern)
            energy.append(self.energy(n_pattern))
        if verbose:
            plt.plot(energy)
            plt.show()
        
        return n_pattern
    
    def async_recall_bias(self, pattern, bias, max_iter=1000, verbose=False):
        """Performs asynchronous recall. Works with only one pattern at once.
        
        Arguments:
            pattern {`np.array`} -- a pattern to recall, stored in a row format
        
        Keyword Arguments:
            max_iter {int} -- Max iterations to perform (default: {1000})
            verbose {bool} -- If True, this will print an image of the pattern every 500 iteration (default: {False})
        
        Returns:
            `np.array` -- A vector in a row format, the retrieved pattern
        """
        
        n_pattern = np.array(pattern, copy=True)
        energy = []
        for j in range(max_iter):
            firing_neuron = np.random.randint(pattern.shape[0])
            new_n_val = 0.5+0.5*sgn(np.dot(pattern, self.W[firing_neuron])-bias)
            n_pattern[firing_neuron] = new_n_val
            if verbose and i%500==0:
                pretty_print(n_pattern)
            energy.append(self.energy(n_pattern))
            if verbose:
                plt.plot(energy)
                plt.show()
        return n_pattern
    

    def energy(self, pattern):
        """Compute the energy of the pattern in the network
        
        Arguments:
            pattern {`np.array`} -- A single pattern to compute the energy for, in a row format     
        
        Returns:
            float -- The energy 
        """
        return - np.dot(np.dot(pattern.T, self.W), pattern)




#%%

# Import patterns
raw_data=open('pict.dat').read() # Raw patterns: string with -1 and +1 \n


patterns_1024=[] # Read patterns in groups of 1024 bits

separate_patterns=np.zeros((32,32,11)) # Save the eleven different patterns in 32x32 images
row_patterns=np.zeros((11,1024)) # Save the eleven different patterns in rows of length 1024

pattern_counter=0 # Counter for the patterns

# Process -1 entrances and read the data
for i in range(len(raw_data)-2):
    if raw_data[i:i+2]=='-1':
        patterns_1024.append(-1)
    elif raw_data[i:i+2]==',1':
        patterns_1024.append(1)
        
    if len(patterns_1024)==1024:
        separate_patterns[:,:,pattern_counter]=np.reshape(np.array(patterns_1024),(32,32))
        row_patterns[pattern_counter,:]=np.array(patterns_1024)
        # Reshape reading array and increase counter
        patterns_1024=[]
        pattern_counter+=1

# Show the 9 patterns
#plt.figure(figsize=(10,10))
#for i in range(11):       
#    plt.subplot(3,4,i+1)
#    plt.imshow(separate_patterns[:,:,i],cmap='gray')



#%%

# 3.5 Capacity
        
# Take the n-th first patterns from images and train the network synchronously
        
erroneous_patterns=[]


for j in range(9):
        
    initial_patterns=row_patterns[0:j+1,:] # Array with j first initial patterns
    
    
    # Train the network through Hebbian learning
    
    w=weight_matrix(initial_patterns)
    
    h=Hopfield(initial_patterns,w)
    
    output=h.sync_recall(initial_patterns,max_iter=1000)
    
    error_vector=np.sum(np.abs(output-initial_patterns),axis=1)

    erroneous_patterns.append(np.count_nonzero(error_vector)) # Add the number of wrong patterns
    
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(range(1,10),erroneous_patterns)
plt.title('Erroneous patterns (images)')
plt.xlabel('Training patterns')
plt.ylabel('Number of erroneous patterns')

plt.subplot(1,2,2)
plt.plot(range(1,10),np.array(erroneous_patterns)*100/np.linspace(1,9,9))
plt.title('Percentage of erroneous patterns (images)')
plt.xlabel('Training patterns')
plt.ylabel('Percentage of erroneous patterns')



#%%
# Train now with random patterns
erroneous_patterns=[]
random_patterns=[]

for j in range(100):
    
    random=np.random.randn(1,1024) # 1024 random numbers from 0 to 1
    
    random[np.where(random<0)]=-1 # -1 assignment
    random[np.where(random>=0)]=1 # 1 assignment
    
    random_patterns.append(random[0])
    random_patterns_array=np.array(random_patterns)
    
    # Train the network through Hebbian learning
    
    w=weight_matrix(random_patterns_array)
    
    h=Hopfield(random_patterns_array,w)
    
    output=h.sync_recall(random_patterns_array,max_iter=1)
    
    error_vector=np.sum(np.abs(output-random_patterns_array),axis=1)
    
    erroneous_patterns.append(np.count_nonzero(error_vector)) # Add the number of wrong patterns

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(range(1,101),erroneous_patterns)
plt.title('Erroneous patterns (random)')
plt.xlabel('Training patterns')
plt.ylabel('Number of erroneous patterns')

plt.subplot(1,2,2)
plt.plot(range(1,101),np.array(erroneous_patterns)*100/np.linspace(1,100,100))
plt.title('Percentage of erroneous patterns (random)')
plt.xlabel('Training patterns')
plt.ylabel('Percentage of erroneous patterns')


#%%

# Train a network with a certain number of nodes with 300 random patterns

nodes=np.linspace(100,1000,10).astype(int)

erroneous=[]

sequence=np.zeros((10,300))

cont=0



for i in nodes:

    random_patterns=np.random.randn(300,i) # Random patterns
    
    
    random_patterns[np.where(random_patterns<-0.5)]=-1 # -1 assignment
    random_patterns[np.where(random_patterns>=-0.5)]=1 # 1 assignment
    
    for j in range(random_patterns.shape[0]):
        
        input_patterns=random_patterns[0:j+1,:]
        
        # Train the network through Hebbian learning
    
        w=weight_matrix(input_patterns)
        
        h=Hopfield(input_patterns,w)
        
        output=h.sync_recall(input_patterns,max_iter=1)
        
        error_vector=np.sum(np.abs(output-input_patterns),axis=1)
        
        erroneous.append(np.count_nonzero(error_vector)*100/error_vector.shape[0]) # Percentage of wrong patterns
        
    print('Completed: {} %'.format(int(i*100/nodes[-1])))
        
    sequence[cont,:]=np.array(erroneous)
    cont+=1
    
    erroneous=[]
    
num_patterns=np.arange(1,random_patterns.shape[0],1)

nodes,num_patterns=np.meshgrid(nodes,num_patterns)

# Plot the surface.

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(nodes, num_patterns, sequence[:,0:-1].T,cmap=cm.coolwarm)
ax.set_xlabel('Nodes')
ax.set_ylabel('Patterns')
ax.set_zlabel('% Error')
ax.set_title('Random training (Bias)')

#%%
plt.figure()
plt.plot(np.arange(1,random_patterns.shape[0],1),sequence[0,0:-1])



#%%

# Noisy patterns

n_nodes=4000 # Pattern length

training=np.random.randn(300,n_nodes) # Random patterns
    
    
training[np.where(training<0)]=-1 # -1 assignment
training[np.where(training>=0)]=1 # 1 assignment

testing=np.copy(training)

noise_level=np.arange(0.2,0.8,0.2)

error=[]

# Training

weights=weight_matrix(training)

for i in noise_level:
    
    random_ind=np.random.randint(low=0,high=int(n_nodes),size=(int(n_nodes-i*n_nodes)))
    
    if np.remainder(random_ind.shape[0],2)!=0:
        random_ind=random_ind[0:-1]

    for j in np.arange(0,random_ind.shape[0],2):
        testing[:,random_ind[j+1]]=training[:,random_ind[j]]
        testing[:,random_ind[j]]=training[:,random_ind[j+1]]
    
    h=Hopfield(testing,weights)
        
    output=h.sync_recall(testing,max_iter=1)
    
    error.append(np.count_nonzero(np.abs(output-training))/(np.prod(training.shape)))
 #%%   
plt.figure()
plt.plot(noise_level*100,error)
plt.xlabel('Noise level (% flipped units)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Error with respect to noise level for random patterns (no self-connections)')
        
    
    
#%%

# 3.6 Sparsity

# Use 100 nodes for 5 and 10% activities (5 and 10 active nodes, respectively)
# Use 500 nodes for 1% activity (5 active nodes)

n_nodes=500
activity=0.01
bias=np.arange(-0.1,0.1,0.01)

max_iter=10000

# Generate sparse patterns
sparse_patterns=np.zeros((20,n_nodes))
for i in range(sparse_patterns.shape[0]):
    active_ind=np.random.randint(low=0,high=int(n_nodes),size=(int(activity*n_nodes)))
    sparse_patterns[i,active_ind]=1
    


erroneous=[] 
sequence=np.zeros((bias.shape[0],sparse_patterns.shape[0]))   
cont_bias=0
# Testing

for i in bias:
    for j in range(sparse_patterns.shape[0]):
        inp=sparse_patterns[0:j+1,:]
        # Training
        biased_weights=activity_weight_matrix(inp,activity)
        # Validation
        h=Hopfield(inp,biased_weights)
        
        out=np.zeros(inp.shape)

        for k in range(j):
            out[k,:]=h.async_recall_bias(inp[k,:], i, max_iter)
        
        error_vector=np.sum(np.abs(out-inp),axis=1)
        
        erroneous.append(np.count_nonzero(error_vector)) # Percentage of wrong patterns
    
        
    sequence[cont_bias,:]=erroneous
    
    erroneous=[]
    
    cont_bias+=1
    
    print('Completed: {} %'.format(cont_bias*100//bias.shape[0]))
    
    
    
    
#%%   
num_patterns=np.arange(1,sparse_patterns.shape[0]+1,1)

aux_bias,aux_num_patterns=np.meshgrid(bias,num_patterns)

# Plot the surface.

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(aux_bias, aux_num_patterns, sequence.T,cmap=cm.coolwarm)
ax.set_xlabel('Bias')
ax.set_ylabel('Patterns')
ax.set_zlabel('Erroneous')
ax.set_title('1% activity')    

