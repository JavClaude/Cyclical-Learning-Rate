from keras import backend as K
from keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np


class CyclicalLearningRate(Callback):
    '''
    Implementation du Cyclical Learning Rate pour Keras
    https://arxiv.org/pdf/1506.01186.pdf
    Auteur: Leslie N. Smith 
    
   
    '''

    def __init__(self, min_lr=float, max_lr=float, stepsize=int, cyclical_type=str, gamma=0.9994):
        '''
        Description
        -----------
        Cyclical Learning Rate implementation for Keras framework

        Parameters
        ----------
        min_lr: float, minimum learning rate boundary
        max_lr: float, maximum learning rate boundary
        stepsize: int, half the cycle length
        cyclicale_type: str
            - triangular: The LR varies linearly between the min_lr and max_lr
            - triangular2: The LR difference (compute_clr) is cut in half at the end of each cycle
            - exp-range: The LR varies between the min_lr and the max_lr and each boundary value declines by an exponential factor: gamma^n°batch
        gamma: float, exponential factor for boundary decay

        Description :
        -------------

        Parameters :
        ------------
            - min_lr: float, valeur minimale du learning rate
            - max_lr: float, valeur maximale du learning rate
            - stepsize: int, moitie du cycle ()
            - cyclicale_type: str
                - triangular: Variation lineaire du lr entre les deux bornes min/max
                - triangular2: The LR difference (compute_clr) is cut in half at the end of each cycle
                - exp_range: le LR varie entre min_lr et max_lr, baisse avec un facteur exponentiel : gamma**n_batch
            - gamma: float, facteur exponentiel regissant la baisse de la frontiere max
            
        '''
        super(CyclicalLearningRate, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.clr = 0
        self.stepsize = stepsize
        self.cyclical_type = cyclical_type
        self.gamma = gamma
        self.iteration = 0
        self.lr_history = []

    def compute_clr(self):
        '''
        Description : 
        -------------
        Main:
            - Definiton du cycle
            - Definition de la position sur le cycle pour chaque iteration
            - Calcul du LR en fonction du type de cycle choisi 
            
        '''
        local_cycle = np.floor(1 + self.iteration / (2 * self.stepsize))

        local_position = np.abs(self.iteration / self.stepsize - 2 * local_cycle + 1)

        if self.cyclical_type == "triangular":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)))

        if self.cyclical_type == "triangular2":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)) * (
                        1 / 2 ** (local_cycle - 1)))

        if self.cyclical_type == "exp_range":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)) * (
                        self.gamma ** self.iteration))

        return local_lr

    def on_train_begin(self, logs):
        '''
        Description :
        -------------
        Methode qui initialise (de maniere forcee) le LR min au premier batch
     
        '''
        if self.iteration == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.compute_clr())

    def on_batch_end(self, batch, logs):
        '''
        Description :
        -------------
        Methode qui permet d'incrementer le LR en fonction des parametres definis par l'utilisateur
        
        '''
        self.iteration += 1

        if self.iteration == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)

        else:
            K.set_value(self.model.optimizer.lr, self.compute_clr())

        self.lr_history.append(self.compute_clr())

    def plot_cycle(self):
        '''
        Description:
        ------------
        Methode qui permet de représenter graphiquement les cycles suivis par le Learning rate

        '''
        plt.plot(self.iteration, self.lr_history)
        plt.title('Cyclical Learning Rate')
        plt.xlabel("Iteration")
        plt.ylabel("LearningRate")
        plt.show()
