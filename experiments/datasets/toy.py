# Imports
import numpy as np
import tensorflow as tf
import math


# Custom function to load toy dataset
def load(name):
    """
    This function creates the toy dataset specified by the `name` input.
    
      - Datasets: sine / sawtooth / square
    """
    
    # Create training signal
    x = np.arange(0, 8, 0.01) * math.pi
    if name == 'sine':
        y = np.sin(x)
    elif name == 'sawtooth':
        y = np.sin(x) - np.sin(2*x)/2 + np.sin(3*x)/3 - np.sin(4*x)/4
    elif name == 'square':
        y = np.sin(x) + np.sin(3*x)/3 + np.sin(5*x)/5 + np.sin(7*x)/7

    # Return data
    return (x, y)