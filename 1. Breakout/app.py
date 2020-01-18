import os
import numpy as np
import tensorflow as tf 

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256, input_dims=(210,160),
                chkpt_dir='/tmp'):
        self.lr = lr
        