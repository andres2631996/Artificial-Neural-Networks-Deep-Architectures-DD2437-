# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:37:20 2019

@author: Usuario
"""

from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet




if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000
    )


#%%
    """ restricted boltzmann machine """

    print("\nStarting a Restricted Boltzmann Machine..")
    en = np.zeros((10,4))
    er = np.zeros((10,4))
    cont = 0
    for i in np.arange(200,600,100):
        rbm = RestrictedBoltzmannMachine(
            ndim_visible=image_size[0] * image_size[1],
            ndim_hidden=i,
            is_bottom=True,
            image_size=image_size,
            is_top=False,
            n_labels=10,
            batch_size=20,
        )
        
        result, final_v_prob, energy, error = rbm.cd1(visible_trainset=train_imgs, epochs=10, lr=1e-2, shuffle=True, verbose=True)
        
        en[:,cont] = energy
        er[:,cont] = error
        
        cont += 1
    #%%
    plt.figure(figsize=(13,5))
    plt.subplot(121)
    plt.plot(range(1,11), en[:,0],'b',label='200 nodes hidden layer')
    plt.plot(range(1,11), en[:,1],'orange',label='300 nodes hidden layer')
    plt.plot(range(1,11), en[:,2],'g',label='400 nodes hidden layer')
    plt.plot(range(1,11), en[:,3],'r',label='500 nodes hidden layer')
    plt.xlabel('Epochs')
    plt.ylabel('Energy')
    plt.title('Energy variation with the number of epochs')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(range(1,11), er[:,0],'b',label='200 nodes hidden layer')
    plt.plot(range(1,11), er[:,1],'orange',label='300 nodes hidden layer')
    plt.plot(range(1,11), er[:,2],'g',label='400 nodes hidden layer')
    plt.plot(range(1,11), er[:,3],'r',label='500 nodes hidden layer')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction loss')
    plt.title('Learning curve')
    plt.legend()

#%%



    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)
    
#%%
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10)
    
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)
    
    #dbn.recognize(train_imgs, train_lbls)

    #dbn.recognize(test_imgs, test_lbls)
    n="rbms"
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, n)
        
#%%

''' fine-tune wake-sleep training: 4 layers '''

dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                    image_size=image_size,
                    n_labels=10,
                    batch_size=10)

acc_greedy_train = []
acc_greedy_test = []
acc_ws_train = []
acc_ws_test = []

for _ in range(10):
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)
    
    a = dbn.recognize(train_imgs, train_lbls,verbose=True)
    b = dbn.recognize(test_imgs, test_lbls,verbose=True)
    
    acc_greedy_train.append(a)
    acc_greedy_test.append(b)
    
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)
    
    
    c = dbn.recognize(train_imgs, train_lbls, verbose=True)
    d = dbn.recognize(test_imgs, test_lbls,verbose=True)
    
    acc_ws_train.append(c)
    acc_ws_test.append(d)
    

mean_a = np.mean(np.array(acc_greedy_train))
mean_b = np.mean(np.array(acc_greedy_test))
mean_c = np.mean(np.array(acc_ws_train))
mean_d = np.mean(np.array(acc_ws_test))

print(mean_a,mean_b,mean_c,mean_d)

std_a = np.std(np.array(acc_greedy_train))
std_b = np.std(np.array(acc_greedy_test))
std_c = np.std(np.array(acc_ws_train))
std_d = np.std(np.array(acc_ws_test))

print(std_a,std_b,std_c,std_d)




#for digit in range(10):
#    digit_1hot = np.zeros(shape=(1,10))
#    digit_1hot[0,digit] = 1
#    dbn.generate(digit_1hot, name="dbn")



#%%
    
    ''' fine-tune wake-sleep training: 3 layers '''
    
dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "top":2000, "lbl":10},
                    image_size=image_size,
                    n_labels=10,
                    batch_size=10)

dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)

dbn.recognize(train_imgs, train_lbls)
dbn.recognize(test_imgs, test_lbls)

dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)



dbn.recognize(train_imgs, train_lbls)
dbn.recognize(test_imgs, test_lbls)


#%%

for digit in range(10):
    digit_1hot = np.zeros(shape=(1,10))
    digit_1hot[0,digit] = 1
    dbn.generate(digit_1hot, name="dbn")
    
    

    
    