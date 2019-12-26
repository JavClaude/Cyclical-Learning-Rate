from keras import backend as K
from keras.callbacks import Callback

import numpy as np


class CyclicalLearningRate(Callback):
    '''
    Cyclical Learning Rate implementation for Keras framework

    Benefits:
        * No need to tune the LR
        * CLR methods require no additional computation (vs Adaptive Learning Rates)
        * CLR method allows to get out of the saddles points plateaus

    https://arxiv.org/pdf/1506.01186.pdf
        Author: Leslie N. Smith

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
            - exp_range: The LR varies between the min_lr and the max_lr and each boundary value declines by an exponential factor: gamma^n°batch
        gamma: float, exponential factor for boundary decay

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
        Description
        -----------
        Compute CLR (main)

        '''
        local_cycle = np.floor(1 + self.iteration / (2 * self.stepsize))

        local_position = np.abs(self.iteration / self.stepsize - 2 * local_cycle + 1)

        if self.cyclical_type == "triangular":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)))

        if self.cyclical_type == "triangular2":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)) * (
                        1 / 2 ** (local_cycle - 1)))

        if self.cyclical_type == "exp-range":
            local_lr = (self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - local_position)) * (
                        self.gamma ** self.iteration))

        return local_lr

    def on_train_begin(self, logs):
        '''
        Description
        -----------
        Set up right LR for training (1st batch)
        '''
        if self.iteration == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.compute_clr())

    def on_batch_end(self, batch, logs):
        self.iteration += 1

        if self.iteration == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)

        else:
            K.set_value(self.model.optimizer.lr, self.compute_clr())

        self.lr_history.append(self.compute_clr())
