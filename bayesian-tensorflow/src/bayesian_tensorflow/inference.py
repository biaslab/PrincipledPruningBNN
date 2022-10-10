# Imports
import tensorflow as tf

# Local functions
from bayesian_tensorflow import losses



# Custom training step function, for Bayes-by-Backprop
@tf.function
def BBB(model, optim, x_batch, y_batch, n_data):
    """
    This function performs gradient descent on a mini-batch of data, when using Bayes-by-Backprop
    (BBB) as the inference method. It uses the Variational Free Energy (VFE) as its loss function.
    
    It takes the BNN model, optimizer, batch data (x and y) and the total data-size as is inputs. 
    It returns the separate VFE terms (i.e. KL-theta, KL-tau, Acc.) of the mini-batch.
    
    When performing the gradient update, the VFE loss is scaled by the batch-size as this results
    is a more stable training procedure. The returned VFE values are not scaled!
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
    
    # Open GradientTape
    with tf.GradientTape() as tape:
        
        # Perform forward pass
        y_pred = model(x_batch, training=True)
        
        # Get KL-losses, scaled to percentage of data
        kl_theta = sum(model.losses) / n_data * b_size
        kl_tau   = model.layers[-1].KL() / n_data * b_size
        
        # Compute accuracy loss
        acc_loss = losses.AccLossBBB(tf.cast(y_batch, dtype=tf.float32), y_pred)
        
        # Full Varational Free Energy loss, scaled down by batch-size
        batch_loss = (kl_theta + kl_tau + acc_loss) / b_size

    # Compute gradients after batch
    grads = tape.gradient(batch_loss, model.trainable_weights)
    
    # Reset gradients with NaN values
    for i in range(len(grads)):
        grads[i] = tf.where(tf.math.is_finite(grads[i]), grads[i], tf.zeros_like(grads[i]))
    
    # Optimize model parameters
    optim.apply_gradients(zip(grads, model.trainable_weights)) 
    
    # Return separate losses
    return kl_theta, kl_tau, acc_loss



# Custom training step function, for Variance Back-Propagation
@tf.function
def VBP(model, optim, x_batch, y_batch, n_data):
    """
    This function performs gradient descent on a mini-batch of data, when using Variance Back-Propagation
    (VBP) as the inference method. It uses the Variational Free Energy (VFE) as its loss function.
    
    It takes the BNN model, optimizer, batch data (x and y) and the total data-size as is inputs. 
    It returns the separate VFE terms (i.e. KL-theta, KL-tau, Acc.) of the mini-batch.
    
    When performing the gradient update, the VFE loss is scaled by the batch-size as this results
    is a more stable training procedure. The returned VFE values are not scaled!
    """
    
    # Get batch-size
    b_size = tf.cast(tf.shape(x_batch)[0], dtype=tf.float32)
    
    # Open GradientTape
    with tf.GradientTape() as tape:
        
        # Perform forward pass
        y_pred = model(x_batch, training=True)
        
        # Get KL-losses, scaled to percentage of data
        kl_theta = sum(model.losses) / n_data * b_size
        kl_tau   = model.layers[-1].KL() / n_data * b_size
        
        # Compute accuracy loss
        acc_loss = losses.AccLossVBP(tf.cast(y_batch, dtype=tf.float32), y_pred)
        
        # Full Varational Free Energy loss, scaled down by batch-size
        batch_loss = (kl_theta + kl_tau + acc_loss) / b_size

    # Compute gradients after batch
    grads = tape.gradient(batch_loss, model.trainable_weights)
    
    # Reset gradients with NaN values
    for i in range(len(grads)):
        grads[i] = tf.where(tf.math.is_finite(grads[i]), grads[i], tf.zeros_like(grads[i]))
    
    # Optimize model parameters
    optim.apply_gradients(zip(grads, model.trainable_weights)) 
    
    # Return separate losses
    return kl_theta, kl_tau, acc_loss