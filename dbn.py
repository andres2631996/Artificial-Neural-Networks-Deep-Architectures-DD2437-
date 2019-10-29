# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:40:55 2019

@author: Usuario
"""

from util import *
from rbm import RestrictedBoltzmannMachine


class DeepBeliefNet:

    """
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf
    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    """

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            "vis--hid": RestrictedBoltzmannMachine(
                ndim_visible=sizes["vis"],
                ndim_hidden=sizes["hid"],
                is_bottom=True,
                image_size=image_size,
                batch_size=batch_size,
            ),
            "hid--pen": RestrictedBoltzmannMachine(
                ndim_visible=sizes["hid"],
                ndim_hidden=sizes["pen"],
                batch_size=batch_size,
            ),
            "pen+lbl--top": RestrictedBoltzmannMachine(
                ndim_visible=sizes["pen"] + sizes["lbl"],
                ndim_hidden=sizes["top"],
                is_top=True,
                n_labels=n_labels,
                batch_size=batch_size,
            ),
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 5

        self.print_period = 100

        return

    def recognize(self, true_img, true_lbl,verbose=False):

        """Recognize/Classify the data into label categories and calculate the accuracy
        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]

        vis = true_img  # visible layer gets the image data

        lbl = (
            np.ones(true_lbl.shape) / 10.0
        )  # start the net by telling you know nothing about labels

        p_h, h = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
        p_h2, h2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(p_h)
        support = np.concatenate((p_h2, lbl),axis=1)
        for _ in range(self.n_gibbs_recog):
            p_h_top, h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(support)
          
            pred, support = self.rbm_stack["pen+lbl--top"].get_v_given_h(p_h_top)

        predicted_lbl = pred[:,-self.rbm_stack["pen+lbl--top"].n_labels:]

        print(
            "accuracy = %.2f%%"
            % (
                100.0
                * np.mean(
                    np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1)
                )
            )
        )
        if verbose==False:
            return
        else:
            return 100.0 * np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))
    
    def generate(self, true_lbl, name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        #records = []
        #fig, ax = plt.subplots(1, 1, figsize=(3, 3))  # ,constrained_layout=True)
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #ax.set_xticks([])
        #ax.set_yticks([])

        lbl = true_lbl
    
        
        pen = np.random.rand(1,self.sizes["pen"])
        
        
        #pen = np.copy(bias_p)
        
        support=np.concatenate((pen,lbl),axis=1)
        
        #support=sample_binary(support)

        for i in range(self.n_gibbs_gener):
            
            prob_top,state_top=self.rbm_stack["pen+lbl--top"].get_h_given_v(support)
            prob_pen_lbl,support=self.rbm_stack["pen+lbl--top"].get_v_given_h(state_top)
            prob_pen_lbl[:, -lbl.shape[1]:]=lbl
            support[:, -lbl.shape[1]:]=lbl
            
            
            
        prob_pen_lbl,state_pen_lbl=self.rbm_stack["pen+lbl--top"].get_v_given_h(state_top)
        state_pen=support[:, :-lbl.shape[1]]
        prob_pen=prob_pen_lbl[:, :-lbl.shape[1]]
        
        prob_hid,state_hid=self.rbm_stack["hid--pen"].get_v_given_h_dir(state_pen)
        prob_vis,state_vis=self.rbm_stack["vis--hid"].get_v_given_h_dir(state_hid)

        #vis = np.random.rand(n_sample, self.sizes["vis"])
        
        # Visualize generated image
        
        plt.figure()
        plt.imshow(np.reshape(prob_vis,self.image_size),cmap='gray')
        plt.title('Generated image')

        #records.append(
        #    [
        #        ax.imshow(
        #            vis.reshape(self.image_size),
        #            cmap="bwr",
        #            vmin=0,
        #            vmax=1,
        #            animated=True,
        #            interpolation=None,
        #        )
        #    ]
        #)

        #anim = stitch_video(fig, records).save(
            #"%s.generate%d.mp4" % (name, np.argmax(true_lbl))
        #)

        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.
        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        except (IOError, OSError):

            print("training vis--hid")
            # CD-1 training for vis--hid
            _ = self.rbm_stack["vis--hid"].cd1(vis_trainset, epochs=n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()
            """
            CD-1 training for hid--pen
            """
            p_h1, h1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
            _ = self.rbm_stack["hid--pen"].cd1(p_h1, epochs=n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """
            CD-1 training for pen+lbl--top
            """
            p_h2, h2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(p_h1)
            # concat input with labels
            h2_lbl = np.concatenate((p_h2, lbl_trainset), axis=1)
            _ = self.rbm_stack["pen+lbl--top"].cd1(h2_lbl, epochs=n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")
        
        
            return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.
        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")

        except (IOError, OSError):

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):
                indexes = np.random.permutation(self.n_samples)
                # we want to do batches !
                for batch_index in np.arange(0, self.n_samples, self.batch_size):
                    vis = vis_trainset[indexes[batch_index:batch_index+self.batch_size]]
                    lbl = lbl_trainset[indexes[batch_index:batch_index+self.batch_size]]
                    # wake-phase : drive the network bottom-to-top using visible and label data

                    # alternating Gibbs sampling in the top RBM : also store neccessary information for learning this RBM
                    # we generate the hidden layer using the dataset
                    p_h_hid, h_hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
                    p_h_pen, h_pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_hid)
                    v_support_0 = np.concatenate((h_pen, lbl), axis=1)
                    # training the top layer (Gibbs Sampling)
                    p_h_top_0, h_top_0 = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_support_0)
                    h_top = np.array(h_top_0,copy=True)
                    for _ in range(self.n_gibbs_wakesleep):
                        # negative phase
                        p_v_support, v_support = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top)
                        # positive phase
                        p_h_top, h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_support)

                    # support, p_support = self.rbm_stack["pen+lbl--top"].cd1(support, epochs=1)
                    p_v_pen, v_pen = p_v_support[:,:self.sizes["pen"]], v_support[:,:self.sizes["pen"]]

                    # sleep phase : from the activities in the top RBM, drive the network top-to-bottom
                    p_v_hid, v_hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(v_pen)
                    p_v_vis, v_vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(v_hid)

                    # predictions : compute generative predictions from wake-phase activations,
                    # and recognize predictions from sleep-phase activations
                    # we want to compare the generation with what we recongnized
                    # earlier
                    pred_p_v_vis, pred_v_vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_hid)
                    pred_p_v_hid, pred_v_hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_pen)

                    pred_p_h_hid, pred_h_hid =  self.rbm_stack["vis--hid"].get_h_given_v_dir(v_vis)
                    pred_p_h_pen, pred_h_pen =  self.rbm_stack["hid--pen"].get_h_given_v_dir(v_hid)
                    # update generative parameters :
                    # here you will only use "update_generate_params" method from rbm class
                    self.rbm_stack["vis--hid"].update_generate_params(h_hid, vis, pred_p_v_vis)
                    self.rbm_stack["hid--pen"].update_generate_params(h_pen, h_hid, pred_p_v_hid)
                    # update parameters of top rbm:
                    # here you will only use "update_params" method from rbm class
                    self.rbm_stack["pen+lbl--top"].update_params(v_support_0, h_top_0, v_support, h_top)
                    # update recognize parameters :
                    # here you will only use "update_recognize_params" method from rbm class
                    self.rbm_stack["vis--hid"].update_recognize_params(vis, h_hid, pred_p_h_hid)
                    pred_h_pen = pred_h_pen[:,:self.sizes["pen"]]
                    self.rbm_stack["hid--pen"].update_recognize_params(h_hid, h_pen, pred_p_h_pen)

                print("epoch=%7d" % it)

            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name, wake_sleep=False):

        self.rbm_stack[name].weight_vh = np.load(
            "%s/rbm.%s.weight_vh.npy" % (loc, name)
        )
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        if wake_sleep==True:
            return self.rbm_stack[name].weight_vh,self.rbm_stack[name].bias_v, self.rbm_stack[name].bias_h
        else:
            return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load(
            "%s/dbn.%s.weight_v_to_h.npy" % (loc, name)
        )
        self.rbm_stack[name].weight_h_to_v = np.load(
            "%s/dbn.%s.weight_h_to_v.npy" % (loc, name)
        )
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save(
            "%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h
        )
        np.save(
            "%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v
        )
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return