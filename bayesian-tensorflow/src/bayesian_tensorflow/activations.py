# Imports
import math
from keras import backend as K
import tensorflow as tf



# ReLU function
@tf.function
def relu_moments(h_mean, h_var):
    """
    This functions computes the first and second (central) moment of a Normal distribution
    passing through a ReLU function.
    
    It takes the mean and variance of the Normal as its inputs, and returns the mean and 
    variance of the resulting output Normal distribution.
    
    The moment are computed using the well-defined moments of a rectified Normal distribution.
    """
        
    # Get std.dev.
    h_std = K.sqrt(h_var)

    # Compute intermediate values
    a_pre = -(h_mean / h_std)
    a     = tf.where(tf.math.is_nan(a_pre), tf.zeros_like(a_pre), a_pre)
    Z     = 0.5 - 0.5 * tf.math.erf(a / tf.math.sqrt(2.))
    phi   = 1./tf.math.sqrt(2*math.pi) * K.exp(-0.5 * K.square(a))

    # Compute mean ...
    y_mean = h_mean * Z + h_std * phi
    # ... and variance
    y_var  = (h_var + K.square(h_mean)) * Z + h_mean * h_std * phi - K.square(y_mean)

    # Return moments
    return y_mean, y_var



# Sigmoid function
@tf.function
def sigmoid_moments(h_mean, h_var):
    """
    This function computed the first and second (central) moment of a Normal distribution
    passing through a Sigmoid function.
    
    It takes the mean and variance of the Normal as its inputs, and returns the mean and 
    variance of the resulting output Normal distribution.
    
    The moment are computed using an approximation of the sigmoid function by means of the
    cumulative distribution function of a Normal distribution.
    """

    # Intermediate value
    t = K.sqrt(1. + math.pi / 8. * K.square(h_var))

    # Compute mean ...
    y_mean = tf.math.sigmoid(h_mean / t)
    # .. and variance
    y_var  = y_mean * (1. - y_mean) * (1. - 1./t)

    # Return moments
    return y_mean, y_var



# Hyperbolic tangent function
@tf.function
def tanh_moments(h_mean, h_var):
    """
    This function computed the first and second (central) moment of a Normal distribution
    passing through a hyperbolic tangent function.
    
    It takes the mean and variance of the Normal as its inputs, and returns the mean and 
    variance of the resulting output Normal distribution.
    
    The moment are computed using a linear transform of the sigmoid function.
    """

    # Use sigmoid moments ...
    s_mean, s_var = sigmoid_moments(2*h_mean, 4*h_var)

    # ... and linear transforms
    y_mean, y_var = 2*s_mean - 1, 4*s_var

    # Return moments
    return y_mean, y_var