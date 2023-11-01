# Imports
import tensorflow as tf

# Local functions
from bayesian_tensorflow import losses



# Custom training step function for Bayes-by-Backprop
@tf.function
def BBB(model, x_batch, y_batch, n_data):
    """
    This function evaluation the Variational Free Energy (VFE) when using the Bayes-by-Backprop (BBB)
    inference method.
    
    It takes the BNN model, batch data (x and y) and the total data-size as is inputs. It returns the
    VFE value of the mini-batch.
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
        
    # Perform forward pass
    y_pred = model(x_batch, training=False)

    # Get KL-losses, scaled to percentage of data
    kl_theta = sum(model.losses) / n_data * b_size
    kl_tau   = model.layers[-1].KL() / n_data * b_size

    # Compute accuracy loss
    acc_loss = losses.AccLossBBB(tf.cast(y_batch, dtype=tf.float32), y_pred)
    
    # Return total VFE loss
    return kl_theta + kl_tau + acc_loss


@tf.function
def BBB_all(model, x_batch, y_batch, n_data):
    """
    This function evaluation the complexity and accuracy of the Variational Free Energy (VFE) when using the Bayes-by-Backprop (BBB)
    inference method.
    
    It takes the BNN model, batch data (x and y) and the total data-size as is inputs. It returns the
    VFE value of the mini-batch.
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
        
    # Perform forward pass
    y_pred = model(x_batch, training=False)

    # Get KL-losses, scaled to percentage of data
    kl_theta = sum(model.losses) / n_data * b_size
    kl_tau   = model.layers[-1].KL() / n_data * b_size

    # Compute accuracy loss
    acc_loss = losses.AccLossBBB(tf.cast(y_batch, dtype=tf.float32), y_pred)
    
    # Return complexity and accuracy loss
    return kl_theta + kl_tau, acc_loss



# Custom training step function, for Variance Back-Propagation
@tf.function
def VBP(model, x_batch, y_batch, n_data):
    """
    This function evaluation the Variational Free Energy (VFE) when using the Variance Back-Propagation
    (VBP) inference method.
    
    It takes the BNN model, batch data (x and y) and the total data-size as is inputs. It returns the
    VFE value of the mini-batch.
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
        
    # Perform forward pass
    y_pred = model(x_batch, training=False)

    # Get KL-losses, scaled to percentage of data
    kl_theta = sum(model.losses) / n_data * b_size
    kl_tau   = model.layers[-1].KL() / n_data * b_size

    # Compute accuracy loss
    acc_loss = losses.AccLossVBP(tf.cast(y_batch, dtype=tf.float32), y_pred)
    
    # Return total VFE loss
    return kl_theta + kl_tau + acc_loss


@tf.function
def VBP_all(model, x_batch, y_batch, n_data):
    """
    This function evaluation the complexity and accuracy of the Variational Free Energy (VFE) when using the Variance Back-Propagation
    (VBP) inference method.
    
    It takes the BNN model, batch data (x and y) and the total data-size as is inputs. It returns the
    VFE value of the mini-batch.
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
        
    # Perform forward pass
    y_pred = model(x_batch, training=False)

    # Get KL-losses, scaled to percentage of data
    kl_theta = sum(model.losses) / n_data * b_size
    kl_tau   = model.layers[-1].KL() / n_data * b_size

    # Compute accuracy loss
    acc_loss = losses.AccLossVBP(tf.cast(y_batch, dtype=tf.float32), y_pred)
    
    # Return complexity and accuracy loss
    return kl_theta + kl_tau, acc_loss