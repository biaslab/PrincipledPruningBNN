# Imports
import math
from keras import backend as K
import tensorflow as tf



# Accuracy loss function for regression models, for Bayes-by-Backprop
@tf.function
def AccLossBBB(y_true, y_pred):
    """
    This function computes the accuracy loss term of the Variational Free Energy (VFE) for the
    Bayes-by-Backprop (BBB) inference method.
    
    It takes the true target value and the model prediction as its inputs.
    """
    
    # Split prediction
    y_samp, alpha, beta = tf.unstack(tf.squeeze(y_pred), 3, axis=-1)
    
    # Compute expected tau values
    tau     = alpha / beta
    log_tau = K.sum(K.mean(tf.math.digamma(alpha) - tf.math.log(beta), axis=0))
    
    # Get output dimension
    M = tf.cast(tf.rank(alpha), dtype=tf.float32)

    # Return accuracy loss
    return 0.5 * K.sum(tau * K.square(y_true - y_samp) + M * K.log(2 * math.pi) - log_tau)



# ACcuracy loss function for regression models, for Variance Back-Propagation
@tf.function
def AccLossVBP(y_true, y_pred):
    """
    This function computes the accuracy loss term of the Variational Free Energy (VFE) for the
    Variance Back-Propagation inference method.
    
    It takes the true target value and the model prediction as its inputs.
    """
    
    # Split prediction
    y_mean, y_var, alpha, beta = tf.unstack(tf.squeeze(y_pred), 4, axis=-1)
    
    # Compute expected values tau
    tau     = alpha / beta
    log_tau = K.sum(K.mean(tf.math.digamma(alpha) - tf.math.log(beta), axis=0))
    
    # Get output dimension
    M = tf.cast(tf.rank(alpha), dtype=tf.float32)
    
    # Return accuracy loss
    return 0.5 * K.sum(tau * (K.square(y_true - y_mean) + y_var) + M * K.log(2 * math.pi) - log_tau)