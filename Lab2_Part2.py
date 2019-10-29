# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:38:55 2019

@author: Usuario
"""

# Code for Part 2 of Lab 2 in ANNDA

# Import necessary packages
import time
start=time.time()

import numpy as np
import matplotlib.pyplot as plt
import operator


# Function for weight updates in SOM algorithm

def neighborhood(pattern,weights,winner,epoch_num,total_epochs,eta,initial_neighbors,circular=False):
    # Get the weight updates for a decreasing neighborhood with the amount of epochs
    increment=np.zeros(weights.shape)
    #points=np.linspace(0,weights.shape[0]-1,weights.shape[0]).astype(int)
    if weights.shape[0]>10:
        inferior=winner-(initial_neighbors-((initial_neighbors-1)/total_epochs)*epoch_num)
        superior=winner+(initial_neighbors-((initial_neighbors-1)/total_epochs)*epoch_num)
    else:
        if epoch_num<int(total_epochs/2):
            inferior=winner-initial_neighbors
            superior=winner+initial_neighbors
        elif epoch_num>int(total_epochs/3) and epoch_num<int(2*total_epochs/3):
            inferior=winner-int(initial_neighbors/2)
            superior=winner+int(initial_neighbors/2)
        else:
            inferior=winner
            superior=winner
    if circular==False:
        if inferior<0:
            inferior=0
        elif superior>weights.shape[0]-1:
            superior=weights.shape[0]-1
        points=np.linspace(inferior,superior-1,superior-inferior).astype(int)
        increment[points,:]=eta*(-weights[points,:]+pattern) # All increments
    if circular==True:
        if superior>=weights.shape[0]:
            points2=np.linspace(inferior,weights.shape[0]-1,weights.shape[0]-inferior).astype(int)
            points1=np.linspace(0,superior-weights.shape[0],superior-weights.shape[0]+1).astype(int)
            points=np.concatenate((points1,points2),axis=0)
            #points=np.linspace(inferior,superior,superior-inferior+1).astype(int)
            increment[points,:]=eta*(-weights[points,:]+pattern) # All increments
        else:
            if superior!=inferior:
                points=np.linspace(inferior,superior,superior-inferior+1).astype(int)
                increment[points,:]=eta*(-weights[points,:]+pattern) # All increments
            else:
                increment[winner,:]=eta*(-weights[winner,:]+pattern) # All increments
    
    
    return increment




# SOM function

def SOM_2D(patterns,weights,total_epochs,eta,initial_neighbors,circular=False,replacement=False):
    # Patterns is a 2D array where each row represents a separate pattern
    # Weights is a 2D array where each row represents the weight vector of one node
    # Total_epochs is the total number of epochs to apply
    # eta is the learning rate
    
    # Get the order of the nodes
    
    order=[] # List of winner nodes at last iteration
    
    for j in range(total_epochs):
    
        for i in range(patterns.shape[0]): # Iterate through all input patterns
            distance=weights-patterns[i,:] # Matrix with all the Euclidean distances
            distance_vector=np.linalg.norm(distance,axis=1) # Vector with the scalar distances
            winner=np.argmin(distance_vector) # Index of closest node to input pattern
            weights+=neighborhood(patterns[i,:],weights,winner,j,total_epochs,eta,initial_neighbors,circular)
            #if j==total_epochs-2 and i==patterns.shape[0]-1:
                #print(weights,patterns)
            if j==total_epochs-1:
                if replacement==True:
                    order.append(winner)
                    weights[winner,:]=np.ones(2)
                else:
                    order.append(winner)
        
        
            
    return order


def neighborhood3D(pattern,weights,winner_node,epoch_num,total_epochs,eta,initial_neighbors,circular=False):
    # Get the weight updates for a decreasing neighborhood with the amount of epochs
    increment=np.zeros(weights.shape)
    #points=np.linspace(0,weights.shape[0]-1,weights.shape[0]).astype(int)
    winner_node=np.array(winner_node)
    if epoch_num<int(total_epochs/4):
        [inf_h,inf_v]=winner_node-np.array([initial_neighbors,initial_neighbors])
        [sup_h,sup_v]=winner_node+np.array([initial_neighbors,initial_neighbors])
    elif epoch_num>int(total_epochs/4) and epoch_num<int(total_epochs/2):
        [inf_h,inf_v]=winner_node-np.array([int(initial_neighbors)/2,int(initial_neighbors/2)])
        [sup_h,sup_v]=winner_node+np.array([int(initial_neighbors)/2,int(initial_neighbors/2)])
    elif epoch_num>int(total_epochs/2) and epoch_num<int(3*total_epochs/4):
        [inf_h,inf_v]=winner_node-np.array([int(initial_neighbors)/3,int(initial_neighbors/3)])
        [sup_h,sup_v]=winner_node+np.array([int(initial_neighbors)/3,int(initial_neighbors/3)])
    else:
        [inf_h,inf_v]=winner_node
        [sup_h,sup_v]=winner_node
    if circular==False:
        if inf_h<0:
            inf_h=0
        if inf_v<0:
            inf_v=0
        if sup_h>=weights.shape[0]:
            sup_h=weights.shape[0]-1
        if sup_v>=weights.shape[1]:
            sup_v=weights.shape[1]-1
        #hor_points=np.linspace(inf_h,sup_h,sup_h-inf_h+1).astype(int) # Horizontal neighbor nodes
        #ver_points=np.linspace(inf_v,sup_v,sup_v-inf_v+1).astype(int) # Vertical neighbor nodes
        increment[int(inf_h):int(sup_h),int(inf_v):int(sup_v),:]=eta*(-weights[int(inf_h):int(sup_h),int(inf_v):int(sup_v),:]+pattern) # All increments
        #gauss=np.exp(-(points-winner)**2/((total_epochs-epoch_num+1)*1.2)) # Gaussian weights
        #distrib=np.dot(increment.T,gauss) # Gaussian distribution for weight update

    return increment


def SOM_3D(patterns,weights,total_epochs,eta,initial_neighbors,circular=False,replacement=False):
    # Patterns is a 2D array where each row represents a separate pattern
    # Weights is a 3D array where each plane represents one attribute
    # Total_epochs is the total number of epochs to apply
    # eta is the learning rate
    
    # Get the order of the nodes
    
    order=np.zeros((patterns.shape[0],2)) # List of winner nodes at last iteration
    
    for j in range(total_epochs):
    
        for i in range(patterns.shape[0]): # Iterate through all input patterns
            distance=weights-patterns[i,:] # Matrix with all the Euclidean distances
            distance_vector=np.linalg.norm(distance,axis=2) # Vector with the scalar distances
            winner=np.argmin(distance_vector) # Index of closest node to input pattern
            winner_node=[winner//weights.shape[0],np.remainder(winner,weights.shape[1])]
            weights+=neighborhood3D(patterns[i,:],weights,winner_node,j,total_epochs,eta,initial_neighbors,circular)
            #if j==total_epochs-2 and i==patterns.shape[0]-1:
                #print(weights,patterns)
            if j==total_epochs-1:
                order[i,:]=winner_node
        
        
            
    return order
        
        

#%%
# 4.1 Topological ordering of animal species

# Import animal data

names=open('animalnames.txt').read().split() # List with animal names
num_animals=len(names) # Number of animals

attributes=open('animalattributes.txt').read().split() # List with attribute names
num_attributes=len(attributes) # Number of attributes

raw_data=open('animals.dat').read() # Raw patterns: string with 0, 1, comas and \n


# Get a vector with just 0s and 1s
patterns=[]
for i in range(len(raw_data)):
    if raw_data[i]=='0':
        patterns.append(0)
    elif raw_data[i]=='1':
        patterns.append(1)

patterns=np.array(patterns) # Transform list of patterns into array
patterns=np.reshape(patterns,(num_animals,num_attributes)) # Reshape pattern array into 2D matrix


# SOM hyperparameters

eta=0.2 # Learning rate
epochs=20 # Epochs

num_nodes=100 # Number of nodes for the map

# Initial weight matrix
weights=np.random.rand(num_nodes,num_attributes)

# Call the SOM function for the specified parameters

order=SOM_2D(patterns,weights,epochs,eta,initial_neighbors=50,circular=False)    # Winning nodes for each case at the last iteration

# Organize names and scores in a dictionary

relation={names[0]:order[0]}

for i in range(1,len(names)):
    relation.update({names[i]:order[i]})

# Sort the dictionary

sorted_relation = sorted(relation.items(), key=operator.itemgetter(1))

print(sorted_relation)

#%%

# 4.2 City ordering

# Obtain the information for the cities
raw_input=open('cities.dat').read().split() # Raw patterns: list of strings and floats

# Keep only those lit elements that can be converted into floats
patterns=[]
# Remove comas from list entries and save floats in a new list
for i in range(len(raw_input)):
    if raw_input[i].find(',')!=-1:
        #raw_input[i][raw_input[i].find(',')]=""
        try:
            patterns.append(float(raw_input[i][0:-2]))
        except ValueError:
            patterns.append(-1)
            
    if raw_input[i].find(';')!=-1:
        #raw_input[i][raw_input[i].find(',')]=""
        try:
            patterns.append(float(raw_input[i][0:-2]))
        except ValueError:
            patterns.append(-1)
        
# Remove -1 entries from list
patterns=np.array(patterns) # Convert to array
keep=np.where(patterns!=-1) # Keep non -1 indexes
patterns=patterns[keep]

# Reshape patterns into 10x2 matrix
patterns=np.reshape(patterns,(10,2))

plt.figure()
plt.scatter(patterns[:,0],patterns[:,1])
plt.title('Original cities')

# Node weights
weights=np.random.rand(patterns.shape[0],patterns.shape[1])


# Hyperparameters

epochs=1000
eta=1



city_order=SOM_2D(patterns,weights,epochs,eta,initial_neighbors=2,circular=True,replacement=True)



print(city_order)

# Plotting resulting path
plt.figure()
plt.scatter(patterns[:,0],patterns[:,1])
for i in range(len(city_order)-1):
    plt.plot(np.linspace(patterns[city_order[i],0],patterns[city_order[i+1],0],100),np.linspace(patterns[city_order[i],1],patterns[city_order[i+1],1],100),'r')

plt.plot(np.linspace(patterns[city_order[-1],0],patterns[city_order[0],0],100),np.linspace(patterns[city_order[-1],1],patterns[city_order[0],1],100),'r')



plt.title('Shortest path between cities')


#%%

# 4.3 Votes of MPs

# Import necessary files

# File with MP names
with open('mpnames.txt') as f:
    mp_names = f.readlines()

num_mps=len(mp_names)

for i in range(num_mps):
    mp_names[i]=mp_names[i][0:-2]

# File with MP parties
with open('mpparty.dat') as f:
    raw_party = f.readlines() 
    
# File with MP sex
with open('mpsex.dat') as f:
    raw_sex = f.readlines()
    
# File with MP district
with open('mpdistrict.dat') as f:
    raw_district = f.readlines()    
    
mp_sex=[]
mp_party=[]
mp_district=[]
mp_party.append(1)
for i in range(len(raw_party)):
    if i<len(raw_sex):
        if raw_sex[i]=='\t 0\n':
            mp_sex.append(0)
        elif raw_sex[i]=='\t 1\n':
            mp_sex.append(1)
    
    if i<len(raw_district):
        mp_district.append(int(raw_district[i][-3:-1]))
    
    if raw_party[i]=='\t   0\n':
        mp_party.append(0)
    elif raw_party[i]=='\t   1\n':
        mp_party.append(1)
    elif raw_party[i]=='\t   2\n':
        mp_party.append(2)
    elif raw_party[i]=='\t   3\n':
        mp_party.append(3)
    elif raw_party[i]=='\t   4\n':
        mp_party.append(4)
    elif raw_party[i]=='\t   5\n':
        mp_party.append(5)
    elif raw_party[i]=='\t   6\n':
        mp_party.append(6)
    elif raw_party[i]=='\t   7\n':
        mp_party.append(7)
    
    
# Import patterns of votes

num_votes=31 # Number of votes per MP

raw_data=open('votes.dat').read() # Raw patterns: list of strings and floats
patterns=[]
for i in range(len(raw_data)):
    if raw_data[i]=='0':
        patterns.append(0)
    elif raw_data[i]=='0.5':
        patterns.append(0.5)
    elif raw_data[i]=='1':
        patterns.append(1)

patterns=np.array(patterns)

patterns=np.reshape(patterns,(num_mps,num_votes))



# Define weight matrix and hyperparameters
weights=np.random.rand(10,10,num_votes)

epochs=1000
eta=1

# Call the SOM algorithm

vote_order=SOM_3D(patterns,weights,epochs,eta,initial_neighbors=2,circular=False,replacement=False)

# Plots for different parties in the 10x10 grid

plt.figure()
plt.xlim(0,10)
plt.ylim(0,10)
for j in range(8):
    for i in range(vote_order.shape[0]):
        if mp_party[i] == j:
            plt.plot(vote_order[i,0], vote_order[i,1], color='#000099', marker='o')
    
    if j!=7:    
        plt.figure()
        plt.xlim(0,10)
        plt.ylim(0,10)


# Plots for the districts 
        
plt.figure()
plt.xlim(0,10)
plt.ylim(0,10)
for j in range(30):
    for i in range(vote_order.shape[0]):
        if mp_district[i] == j:
            plt.plot(vote_order[i,0], vote_order[i,1], color='r', marker='o')
    
    if j!=29:    
        plt.figure()
        plt.xlim(0,10)
        plt.ylim(0,10)
        
# Plots for the sex 
        
plt.figure()
plt.xlim(0,10)
plt.ylim(0,10)
for j in range(2):
    for i in range(vote_order.shape[0]):
        if mp_sex[i] == j:
            plt.plot(vote_order[i,0], vote_order[i,1], color='g', marker='o')
    
    if j!=1:    
        plt.figure()
        plt.xlim(0,10)
        plt.ylim(0,10)
        

end=time.time()

print(end-start)