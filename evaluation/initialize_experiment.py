"""
Systematic experimental initialization.
"""

import numpy as np
import torch
import random


def initialize_random_seed(seed=3):
    """
    Initialize random seeds for an experiment.
    
    Parameters:
        seed (int): Random seed to use
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
