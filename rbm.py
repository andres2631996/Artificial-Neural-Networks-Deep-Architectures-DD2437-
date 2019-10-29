# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:40:07 2019

@author: Usuario
"""

from util import *


class RestrictedBoltzmannMachine:
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    def __init__(
        self,
        ndim_visible,
        ndim_hidden,
        is_bottom=False,
        image_size=[28, 28],
        is_top=False,
        n_labels=10,
        batch_size=10,
    ):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom:
            self.image_size = image_size

        self.is_top = is_top

        if is_top:
            self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)
        )

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 5000

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(
                0, self.ndim_hidden, 25
            ),  # pick some random hidden units
        }

        return

    def cd1(self, visible_trainset, epochs=10, lr=0.01, shuffle=False, verbose=False):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        def _gibbs_sampling(v_0):
            """
            Helper function to performs gibbs sampling
            """
            # positive phase
            p_h_0, h_0 = self.get_h_given_v(v_0)

            # negative phase
            p_v_1, v_1 = self.get_v_given_h(h_0)

            p_h_1, h_1 = self.get_h_given_v(v_1)

            # updating parameters
            self.update_params(v_0, p_h_0, p_v_1, p_h_1, lr)
            
            return

        def _reconstruct_visible(trainset, batch_size=20, energy = False):
            """
            Helper function to recompute the V in batch
            """
            n_samples = trainset.shape[0]
            P_V = np.zeros((n_samples, self.ndim_visible))
            V = np.zeros((n_samples, self.ndim_visible))
            P_H = np.zeros((n_samples, self.ndim_hidden))
            H = np.zeros((n_samples, self.ndim_hidden))
            indexes = np.arange(len(trainset))
            E=[]
            for idx in indexes[::batch_size]:
                p_h, h = self.get_h_given_v(trainset[idx : idx + batch_size])
                p_v, v = self.get_v_given_h(h)
                P_V[idx : idx + batch_size] = p_v
                V[idx : idx + batch_size] = v
                
                if energy== True:
                    batch_energy = np.sum(np.matmul(h,np.matmul(self.weight_vh.T,v.T)),axis=1) 
                    + np.matmul(h,self.bias_h) + np.matmul(v,self.bias_v)
                    
                    final_energy = np.sum(batch_energy,axis=0)
                    E.append(final_energy)
                    
            remainder_idx = n_samples % batch_size
            if remainder_idx > 0:
                p_h, h = self.get_h_given_v(trainset[-remainder_idx:])
                p_v, v = self.get_v_given_h(h)
                P_V[idx : idx + batch_size] = p_v
                V[idx : idx + batch_size] = v
                
                if energy== True:
                    batch_energy = np.sum(np.matmul(h,np.matmul(self.weight_vh.T,v.T)),axis=1) 
                    + np.matmul(h,self.bias_h) + np.matmul(v,self.bias_v)
                    
                    final_energy = np.sum(batch_energy,axis=0)
                    E.append(final_energy)
                    
            if energy==True:
                return P_V, V, sum(E)
            
            else:
                return P_V, V

        print("learning CD1")

        n_samples = visible_trainset.shape[0]
        n_batches = n_samples/self.batch_size
        # n_batches= np.ceil(n_samples/self.batch_size)
        if shuffle:
            indexes = np.random.permutation(n_samples)
        else:
            indexes = np.arange(n_samples)
        
        if n_batches is not None:
            max_index=self.batch_size*n_batches
        else:
            max_index=n_samples
            
        if verbose:
            error=[] # List to save reconstruction loss
            energy=[]
        for epoch in range(epochs):
            for batch_index in np.arange(0, n_samples, self.batch_size):
                # random sampling for minibatches
                v_0 = visible_trainset[
                    indexes[batch_index : batch_index + self.batch_size]
                ]
                _gibbs_sampling(v_0)
            remainder_idx = n_samples % self.batch_size
            if remainder_idx > 0:
                v_0 = visible_trainset[indexes[-remainder_idx:]]
                _gibbs_sampling(v_0)

            if verbose:
                # print progress
                _, V,E = _reconstruct_visible(visible_trainset,self.batch_size,energy=True)
                energy.append(E/n_samples)
                error.append(np.linalg.norm(V - visible_trainset))
                if np.remainder(epoch, 1) == 0:
                    if self.is_bottom:
                        viz_rf(
                            weights=self.weight_vh[:, self.rf["ids"]].reshape(
                                (self.image_size[0], self.image_size[1], -1)
                            ),
                            it=epoch + 1,
                            grid=self.rf["grid"],
                        )
                    print(
                        "epoch: %7d recon_loss=%4.4f"
                        % (epoch + 1, np.linalg.norm(V - visible_trainset))
                    )
        final_v_prob, result = _reconstruct_visible(visible_trainset)
        if verbose:
            plt.figure(figsize=(5, 5))
            perm = np.random.choice(len(result), 4)
            plt_idx = 421
            for imgs in zip(final_v_prob[perm], visible_trainset[perm]):
                r, gt = imgs[0], imgs[1]
                plt.subplot(plt_idx)
                plt_idx += 1
                plt.imshow(np.reshape(r, (28, 28)), cmap="gray")
                plt.subplot(plt_idx)
                plt_idx += 1
                plt.imshow(np.reshape(gt, (28, 28)), cmap="gray")
            
            plt.figure(figsize=(15,5))
            plt.subplot(121)
            plt.plot(range(1,epochs+1),error)
            plt.xlabel('Epochs')
            plt.ylabel('Reconstruction Error')
            plt.title('Learning Curve for RBM'.format(self.ndim_hidden))
            
            plt.subplot(122)
            plt.plot(range(1,epochs+1),energy)
            plt.xlabel('Epochs')
            plt.ylabel('Energy')
            plt.title('Average energy per sample evolution')
        plt.show()
        if verbose==True:
            return result, final_v_prob, energy, error
        else:
            return result, final_v_prob

    def update_params(self, v_0, h_0, v_k, h_k, lr=0.01,wake_sleep=False,weight_vh=None,bias_v=None, bias_h=None):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """
        pos = np.tensordot(v_0, h_0, axes=((0), (0)))
        neg = np.tensordot(v_k, h_k, axes=((0), (0)))

        self.delta_bias_v = np.sum(v_0 - v_k, axis=0)
        self.delta_weight_vh = pos - neg
        self.delta_bias_h = np.sum(h_0 - h_k, axis=0)
        
        if wake_sleep==False:
            self.bias_v += (lr) * self.delta_bias_v
            self.weight_vh += (lr) * self.delta_weight_vh
            self.bias_h += (lr) * self.delta_bias_h
        
        else:
            bias_v += (lr) * self.delta_bias_v
            weight_vh += (lr) * self.delta_weight_vh
            bias_h += (lr) * self.delta_bias_h
            
            self.bias_v=bias_v
            self.weight_vh=weight_vh
            self.bias_h=bias_h

        return

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]
        p_h = sigmoid(np.matmul(visible_minibatch, self.weight_vh) + self.bias_h)
        h = sample_binary(p_h)
        # h  = (p_h >=random_h).astype(int)
        return p_h, h

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None
        n_samples = hidden_minibatch.shape[0]
        
        support = np.matmul(hidden_minibatch, self.weight_vh.T) + self.bias_v
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            p_data = sigmoid(support[:, :-self.n_labels])
            p_labels = softmax(support[:, -self.n_labels:])
            p_v=np.concatenate((p_data,p_labels),axis=1)

        else:

            p_v = sigmoid(support)
        
        v = sample_binary(p_v)

        return p_v, v

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self,wake_sleep=False):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None
        
        if wake_sleep==True:
            return self.weight_v_to_h,self.weight_h_to_v

    #%%
    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        p_h = sigmoid(np.matmul(visible_minibatch, self.weight_v_to_h) + self.bias_h)
        h = sample_binary(p_h)
        
        return (
            p_h,
            h,
        )

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]
        
        support = np.matmul(hidden_minibatch, self.weight_h_to_v) + self.bias_v

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            p_data = sigmoid(support[:, :-self.n_labels])
            p_labels = softmax(support[:, -self.n_labels:])
            p_v = np.concatenate((p_data,p_labels),axis=1)

        else:

            p_v = sigmoid(support)
        
        v = sample_binary(p_v)

        return (
            p_v,
            v,
        )

    def update_generate_params(self, inps, trgs, preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        
        #self.delta_weight_h_to_v = np.matmul(inps.T,(trgs-preds))
        #self.delta_bias_v = np.sum(trgs-preds,axis=0)

        #weight_h_v += (1/inps.shape[0])*self.delta_weight_h_to_v
        #bias_v += (1/inps.shape[0])*self.delta_bias_v
        
        #self.weight_h_to_v=weight_h_v
        #self.bias_v=bias_v
        
        support = np.tensordot(inps, trgs-preds, axes=((0), (0)))
        self.weight_h_to_v += self.learning_rate*support
        
        self.bias_v += self.learning_rate*np.sum(trgs - preds, axis=0)
        
        return

    def update_recognize_params(self, inps, trgs, preds):
        

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        #self.delta_weight_v_to_h = np.matmul(inps.T,(trgs-preds))
        #self.delta_bias_h = np.sum(trgs-preds,axis=0)
        
        #weight_v_h += (1/inps.shape[0])*self.delta_weight_v_to_h
        #bias_h = (1/inps.shape[0])*self.delta_bias_h

        #self.weight_v_to_h = weight_v_h
        #self.bias_h = bias_h
        
        
        support = np.tensordot(inps, trgs-preds, axes=((0), (0)))
        self.weight_v_to_h += (self.learning_rate*support)
        
        self.bias_h += self.learning_rate*np.sum(trgs - preds, axis=0)
        
        

        return
