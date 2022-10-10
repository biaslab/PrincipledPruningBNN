# Imports
import math
from keras import backend as K
from keras import initializers
import tensorflow as tf

# Local functions
from bayesian_tensorflow import activations



# Dense layer
class DenseVBP(tf.keras.layers.Layer):
    """
    Variational fully connected layer (dense), following Variance Back-Propagation (VBP).
    
    It takes the number of units as its input, all other inputs are optional.
    """
    
    def __init__(self, 
                 units,                     # number of output features
                 is_input   =  False,       # if layer is input layer
                 is_output  =  False,       # if layer is output layer
                 data_var   =  1e-3,        # initial value for data variance
                 prior_var  =  1.,          # prior variance of parameters
                 std_dev    =  0.01,        # standard deviation of initializer
                 init       = 'prior',      # manner in which params are initialized
                 seed       =  None,        # seed for (param) initialization
                 **kwargs):
        
        # Copy inputs ...
        self.units     = units
        self.is_input  = is_input
        self.is_output = is_output
        self.data_var  = data_var
        self.prior_var = prior_var
        self.std_dev   = std_dev
        self.init      = init
        # ... and set seed
        if seed is not None: 
            tf.random.set_seed(seed)
            
        # Other args
        super().__init__(**kwargs)
    
    
    # Standard function to return output shape
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
    
    
    # Standard function to create layer parameters
    def build(self, input_shape):

        # Initializer
        if self.init == 'prior':      # 'prior' is (sampled around) the prior
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(self.prior_var) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'he':       # 'he' uses mean and variance from HeNormal
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(2. / input_shape[1]) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'glorot':   # 'glorot' uses mean and variance from GlorotNormal
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(2. / (input_shape[1] + self.units)) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'paper':    # 'paper' follows Haussmann et al. (2019)
            self.init_mu   = initializers.HeNormal()
            self.init_rho2 = initializers.normal(mean=-9., stddev=1e-3)
        elif self.init == 'tf':       # 'tf' follows the TensorFlow implementation
            self.init_mu   = initializers.normal(mean=0., stddev=0.1)
            self.init_rho2 = initializers.normal(mean=-6., stddev=0.1)
        
        # Weight matrix 'W', also called kernel
        self.kernel_mu   = self.add_weight(name='kernel_mu', shape=(input_shape[1], self.units),
                                           initializer=self.init_mu, trainable=True)
        self.kernel_rho2 = self.add_weight(name='kernel_rho2', shape=(input_shape[1], self.units),
                                           initializer=self.init_rho2, trainable=True)
        
        # Bias vector 'b'
        self.bias_mu     = self.add_weight(name='bias_mu', shape=(self.units,),
                                           initializer=self.init_mu, trainable=True)
        self.bias_rho2   = self.add_weight(name='bias_rho2', shape=(self.units,),
                                           initializer=self.init_rho2, trainable=True)
        
        # Add KL-divergence loss
        self.add_loss(lambda: self.KL())
        
        # Create masks for pruning
        self.kernel_mask = tf.ones_like(self.kernel_mu)
        self.bias_mask   = tf.ones_like(self.bias_mu)
        
        # Super build function
        super().build(input_shape)
    
    
    # Standard function to compute output
    def call(self, inputs, **kwargs):
        
        # If input layer, create variance
        if self.is_input:
            x_mean, x_var = inputs, self.data_var * tf.ones_like(inputs)
        # Else, split inputs
        else:
            x_mean, x_var = tf.unstack(inputs, axis=-1)
        
        # Gather posterior parameters
        w_mean, b_mean = self.kernel_mu, self.bias_mu
        w_var,  b_var  = K.softplus(self.kernel_rho2), K.softplus(self.bias_rho2)
        
        # Compute E[h] = E[W]*E[x] + E[b]
        h_mean = K.dot(x_mean, w_mean) + b_mean
        
        # Compute Var[h] = Var[x]*(E[W]^2 + Var[W]) + E[x]^2*Var[W] + Var[b]
        h_var = K.dot(x_var, (K.square(w_mean) + w_var)) + K.dot(K.square(x_mean), w_var) + b_var
        
        # Return just output ...
        if self.is_output:
            # i.e. E[h] and Var[h]
            return tf.stack([h_mean, h_var], axis=-1)
        
        # ... or return with ReLU activation function
        else:            
            # i.e. E[ReLU(h)] and Var[ReLU(h)]
            y_mean, y_var = activations.relu_moments(h_mean, h_var)
            return tf.stack([y_mean, y_var], axis=-1)
    
    
    # Custom function to compute KL-divergence loss of layer
    def KL(self):
        
        # Kernel
        w_mean = self.kernel_mu
        w_var  = K.softplus(self.kernel_rho2)
        w_vals = (w_var + K.square(w_mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(w_var)
        KL_w   = 0.5 * K.sum(tf.boolean_mask(w_vals, tf.math.is_finite(w_vals)))
        
        # Bias
        b_mean = self.bias_mu
        b_var  = K.softplus(self.bias_rho2)
        b_vals = (b_var + K.square(b_mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(b_var)
        KL_b   = 0.5 * K.sum(tf.boolean_mask(b_vals, tf.math.is_finite(b_vals)))
        
        # Return sum of KLs
        return KL_w + KL_b
    
    
    # Custom function for compression based on BMR
    def compress(self, red_var=1e-16):
        
        # Kernel matrix
        w_mean = self.kernel_mu
        w_rho2 = self.kernel_rho2
        w_var  = K.softplus(w_rho2)
        # Compute BMR values
        BMR_w = self.BMR(w_mean, w_var, red_var)
        # Compress parameters with dVFE <= 0
        self.kernel_mu.assign(tf.where(BMR_w<=0, tf.zeros_like(w_mean), w_mean))
        self.kernel_rho2.assign(tf.where(BMR_w<=0, -1e5*tf.ones_like(w_rho2), w_rho2))
        # Update kernel mask
        self.kernel_mask = tf.where(BMR_w<=0, tf.zeros_like(w_mean), self.kernel_mask)
        
        # Bias vector
        b_mean = self.bias_mu
        b_rho2 = self.bias_rho2
        b_var  = K.softplus(b_rho2)
        # Compute BMR values
        BMR_b = self.BMR(b_mean, b_var, red_var)
        # Compress parameters with dVFE <= 0
        self.bias_mu.assign(tf.where(BMR_b<=0, tf.zeros_like(b_mean), b_mean))
        self.bias_rho2.assign(tf.where(BMR_b<=0, -1e5*tf.ones_like(b_rho2), b_rho2))
        # Update bias mask
        self.bias_mask = tf.where(BMR_b<=0, tf.zeros_like(b_mean), self.bias_mask)
        
        
    # Custom function to compute BMR values
    def BMR(self, mean, var, red_var):
        
        # Compute intermediate values
        Pi_i = 1. / red_var
        P_f  = 1. / var
        P_i  = P_f + Pi_i - 1. / self.prior_var
        mu_i = P_f * mean / P_i
        
        # Return BMR values
        return 0.5 * ((mean**2 * P_f - mu_i**2 * P_i) - K.log(Pi_i * P_f / P_i * self.prior_var))
    
    
    # Custom function to reset model parameters
    def param_reset(self):
        
        # Kernel matrix
        w_mean = self.kernel_mu
        w_rho2 = self.kernel_rho2
        # Reset kernel
        self.kernel_mu.assign(tf.where(self.kernel_mask==0, tf.zeros_like(w_mean), w_mean))
        self.kernel_rho2.assign(tf.where(self.kernel_mask==0, -1e5*tf.ones_like(w_rho2), w_rho2))
        
        # Bias vector
        b_mean = self.bias_mu
        b_rho2 = self.bias_rho2
        # Reset bias
        self.bias_mu.assign(tf.where(self.bias_mask==0, tf.zeros_like(b_mean), b_mean))
        self.bias_rho2.assign(tf.where(self.bias_mask==0, -1e5*tf.ones_like(b_rho2), b_rho2))



# Custom layer to add Gamma random variable for precision
class GammaVBP(tf.keras.layers.Layer):
    """
    Dummy layer for adding an alpha and beta parameter of a Gamma distribution to a BNN. 
    
    Allows for joint optimization of posterior precision parameter(s).
    """
    
    def __init__(self,
                 units,                     # number of output features
                 alpha = 1.,                # initial value for alpha
                 beta  = 1.,                # initial value for beta
                 **kwargs):
        
        # Set units
        self.units = units
        
        # Set initial alpha and beta value
        self.alpha_init = initializers.constant(K.log(alpha))
        self.beta_init  = initializers.constant(K.log(beta))
            
        # Other args
        super().__init__(**kwargs)
    
    
    # Standard function to return output shape
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
    
    
    # Standard function to create layer parameters
    def build(self, input_shape):
        
        # Add (log) alpha and beta parameters
        self.log_alpha = self.add_weight(name='log_alpha', shape=(self.units,),
                                         initializer=self.alpha_init, trainable=True)
        self.log_beta  = self.add_weight(name='log_beta', shape=(self.units,),
                                         initializer=self.beta_init, trainable=True)
        
        # Super build function
        super().build(input_shape)
    
    
    # Standard function to compute output
    def call(self, inputs, **kwargs):
        
        # Split inputs
        y_mean, y_var = tf.unstack(inputs, 2, axis=-1)
        
        # Extend alpha and beta to match inputs size
        alpha = K.exp(self.log_alpha) * tf.ones_like(y_mean)
        beta  = K.exp(self.log_beta) * tf.ones_like(y_mean)
        
        # Return inputs incl. alpha and beta
        return tf.stack([y_mean, y_var, alpha, beta], axis=-1)
    
    
    # Custom function for KL-divergence
    def KL(self):
        
        # Get alpha and beta
        alpha = K.exp(self.log_alpha)
        beta  = K.exp(self.log_beta)
        
        # Return KL-divergence
        return K.sum((alpha - 1) * tf.math.digamma(alpha) - tf.math.lgamma(alpha) + tf.math.log(beta) + alpha * ((1 - beta) / beta))



# GRU cell (i.e. layer)
class GRUCellVBP(tf.keras.layers.Layer):
    """
    Variational Gated Recurrent Unit (GRU), following Variance Back-Propagation (VBP). 
    
    It takes the number of units as its input, all other inputs are optional.
    """
    
    def __init__(self,
                 units,                     # number of output features
                 is_input   =  False,       # if layer is input layer
                 data_var   =  1e-3,        # initial value for data variance
                 prior_var  =  1.,          # prior variance of parameters
                 std_dev    =  0.01,        # standard deviation of initializer
                 init       = 'prior',      # manner in which params are initialized
                 seed       =  None,        # seed for (weight) initialization
                 **kwargs):
        
        # Copy inputs and set seed
        self.units      = 3*units
        self.state_size = 2*units
        self.is_input   = is_input
        self.data_var   = data_var
        self.prior_var  = prior_var
        self.std_dev    = std_dev
        self.init       = init
        # ... and set seed
        if (init is not None):
            tf.random.set_seed(seed)
            
        # Other args
        super().__init__(**kwargs)
        
        
    # Standard function to return output shape
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
    
    
    # Standard function to create layer parameters
    def build(self, input_shape):
        
        # Initializer
        if self.init == 'prior':      # 'prior' is (sampled around) the prior
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(self.prior_var) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'he':       # 'he' uses mean and variance from HeNormal
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(2. / input_shape[1]) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'glorot':   # 'glorot' uses mean and variance from GlorotNormal
            self.init_mu   = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho2 = initializers.normal(mean=K.log(K.exp(2. / (input_shape[1] + self.units)) - 1.), 
                                                 stddev=self.std_dev)
        elif self.init == 'paper':    # 'paper' follows Haussmann et al. (2019)
            self.init_mu   = initializers.HeNormal()
            self.init_rho2 = initializers.normal(mean=-9., stddev=1e-3)
        elif self.init == 'tf':       # 'tf' follows the TensorFlow implementation
            self.init_mu   = initializers.normal(mean=0., stddev=0.1)
            self.init_rho2 = initializers.normal(mean=-6., stddev=0.1)
        
        # Kernel matrix
        self.W_mu   = self.add_weight(name='W_mu', shape=(input_shape[1], self.units),
                                      initializer=self.init_mu, trainable=True)
        self.W_rho2 = self.add_weight(name='W_rho2', shape=(input_shape[1], self.units),
                                      initializer=self.init_rho2, trainable=True)
        # Hidden matrix
        self.U_mu   = self.add_weight(name='U_mu', shape=(int(self.state_size/2), self.units),
                                      initializer=self.init_mu, trainable=True)
        self.U_rho2 = self.add_weight(name='U_rho2', shape=(int(self.state_size/2), self.units),
                                      initializer=self.init_rho2, trainable=True)
        # Bias vector
        self.b_mu   = self.add_weight(name='b_mu', shape=(self.units,),
                                      initializer=self.init_mu, trainable=True)
        self.b_rho2 = self.add_weight(name='b_rho2', shape=(self.units,),
                                      initializer=self.init_rho2, trainable=True)
        
        # Add KL-divergence loss
        self.add_loss(lambda: self.KL())
        
        # Create masks for pruning
        self.W_mask = tf.ones_like(self.W_mu)
        self.U_mask = tf.ones_like(self.U_mu)
        self.b_mask = tf.ones_like(self.b_mu)
        
        # Super build function
        super().build(input_shape)
    
    
    # Standard function to compute output
    def call(self, inputs, states, **kwargs):
        
        # If input layer, create variance
        if self.is_input:
            x_mean, x_var = inputs, self.data_var * tf.ones_like(inputs)
        # Else, split inputs
        else:
            x_mean, x_var = tf.unstack(inputs, axis=-1)
            
        # Split states
        h_min1_mean, h_min1_var = tf.split(states[0], 2, axis=1)
        
        # Split means ...
        W_r_mu, W_u_mu, W_h_mu = tf.split(self.W_mu, 3, axis=1)
        U_r_mu, U_u_mu, U_h_mu = tf.split(self.U_mu, 3, axis=1)
        b_r_mu, b_u_mu, b_h_mu = tf.split(self.b_mu, 3, axis=0)
        # ... and variances
        W_r_var, W_u_var, W_h_var = tf.split(K.softplus(self.W_rho2), 3, axis=1)
        U_r_var, U_u_var, U_h_var = tf.split(K.softplus(self.U_rho2), 3, axis=1)
        b_r_var, b_u_var, b_h_var = tf.split(K.softplus(self.b_rho2), 3, axis=0)
            
        # Reset gate
        r_mean = b_r_mu + K.dot(x_mean, W_r_mu) + K.dot(h_min1_mean, U_r_mu)
        r_var  = b_r_var + K.dot(x_var, K.square(W_r_mu) + W_r_var) + K.dot(K.square(x_mean), W_r_var) + \
                 K.dot(h_min1_var, K.square(U_r_mu) + U_r_var) + K.dot(K.square(h_min1_mean), U_r_var)
        # Update gate
        u_mean = b_u_mu + K.dot(x_mean, W_u_mu) + K.dot(h_min1_mean, U_u_mu)
        u_var  = b_u_var + K.dot(x_var, K.square(W_u_mu) + W_u_var) + K.dot(K.square(x_mean), W_u_var) + \
                 K.dot(h_min1_var, K.square(U_r_mu) + U_u_var) + K.dot(K.square(h_min1_mean), U_u_var)
        
        # Sigmoid activations
        r_mean, r_var = activations.sigmoid_moments(r_mean, r_var)
        u_mean, u_var = activations.sigmoid_moments(u_mean, u_var)
        
        # Intermediate variance, i.e. Var[r * h]
        int_var = r_var * (K.square(h_min1_mean) * h_min1_var) + h_min1_var * K.square(r_mean)
        # Hidden unit pre
        h_pre_mean = b_h_mu + K.dot(x_mean, W_h_mu) + K.dot(r_mean * h_min1_mean, U_h_mu)
        h_pre_var  = b_h_var + K.dot(x_var, K.square(W_h_mu) + W_h_var) + K.dot(K.square(x_mean), W_h_var) + \
                     K.dot(int_var, K.square(U_h_mu) + U_h_var) + K.dot(K.square(r_mean * h_min1_mean), U_h_var)
        
        # Tanh activation
        h_pre_mean, h_pre_var = activations.tanh_moments(h_pre_mean, h_pre_var)
        
        # Hidden unit final
        h_mean = u_mean * h_min1_mean + (1. - u_mean) * h_pre_mean
        h_var  = u_var * (K.square(h_min1_mean) + h_min1_var) + K.square(u_mean) * h_min1_var + \
                 u_var * (K.square(h_pre_mean) + h_pre_var) + K.square(u_mean) * h_pre_var
        
        # Stack outputs and concat states ...
        outputs = tf.stack([h_mean, h_var], axis=-1)
        states  = tf.concat([h_mean, h_var], axis=1)
        
        # ... and return
        return outputs, [states]
    
    
    # Custom function to compute KL-divergence values given mean and std.dev.
    def kl_value(self, mean, var):
        
        # KL-divergence values
        KL = (var + K.square(mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(var)
        
        # Return filtered values
        return 0.5 * K.sum(tf.boolean_mask(KL, tf.math.is_finite(KL)))
    
    
    # Custom function to compute total KL-divergence loss of layer
    def KL(self):
        
        # Get all variances
        W_var, U_var, b_var = K.softplus(self.W_rho2), K.softplus(self.U_rho2), K.softplus(self.b_rho2)
        
        # Return values
        return self.kl_value(self.W_mu, W_var) + self.kl_value(self.U_mu, U_var) + self.kl_value(self.b_mu, b_var)
    
    
    # Custom function for compression based on BMR
    def compress(self, red_var=1e-16):
        
        # Kernel matrix
        w_mean = self.W_mu
        w_rho2 = self.W_rho2
        w_var  = K.softplus(w_rho2)
        # Compute BMR values
        BMR_w = self.BMR(w_mean, w_var, red_var)
        # Compress parameters with dVFE <= 0
        self.W_mu.assign(tf.where(BMR_w<=0, tf.zeros_like(w_mean), w_mean))
        self.W_rho2.assign(tf.where(BMR_w<=0, -1e5*tf.ones_like(w_rho2), w_rho2))
        # Update kernel mask
        self.W_mask = tf.where(BMR_w<=0, tf.zeros_like(w_mean), self.W_mask)
        
        # Hidden matrix
        u_mean = self.U_mu
        u_rho2 = self.U_rho2
        u_var  = K.softplus(u_rho2)
        # Compute BMR values
        BMR_u = self.BMR(u_mean, u_var, red_var)
        # Compress parameters with dVFE <= 0
        self.U_mu.assign(tf.where(BMR_u<=0, tf.zeros_like(u_mean), u_mean))
        self.U_rho2.assign(tf.where(BMR_u<=0, -1e5*tf.ones_like(u_rho2), u_rho2))
        # Update hidden mask
        self.U_mask = tf.where(BMR_u<=0, tf.zeros_like(u_mean), self.U_mask)
        
        # Bias
        b_mean = self.b_mu
        b_rho2 = self.b_rho2
        b_var  = K.softplus(b_rho2)
        # Compute BMR values
        BMR_b = self.BMR(b_mean, b_var, red_var)
        # Compress parameters with dVFE <= 0
        self.b_mu.assign(tf.where(BMR_b<=0, tf.zeros_like(b_mean), b_mean))
        self.b_rho2.assign(tf.where(BMR_b<=0, -1e5*tf.ones_like(b_rho2), b_rho2))
        # Update bias mask
        self.b_mask = tf.where(BMR_b<=0, tf.zeros_like(b_mean), self.b_mask)
        
        
    # Custom function to compute BMR values
    def BMR(self, mean, var, red_var):
        
        # Compute intermediate values
        Pi_i = 1. / red_var
        P_f  = 1. / var
        P_i  = P_f + Pi_i - 1. / self.prior_var
        mu_i = P_f * mean / P_i
        
        # Return BMR values
        return 0.5 * ((mean**2 * P_f - mu_i**2 * P_i) - K.log(Pi_i * P_f / P_i * self.prior_var))
    
    
    # Custom function to reset model parameters
    def param_reset(self):
        
        # Kernel matrix
        w_mean = self.W_mu
        w_rho2 = self.W_rho2
        # Reset kernel
        self.W_mu.assign(tf.where(self.W_mask==0, tf.zeros_like(w_mean), w_mean))
        self.W_rho2.assign(tf.where(self.W_mask==0, -1e5*tf.ones_like(w_rho2), w_rho2))
        
        # Hidden matrix
        u_mean = self.U_mu
        u_rho2 = self.U_rho2
        # Reset hidden
        self.U_mu.assign(tf.where(self.U_mask==0, tf.zeros_like(u_mean), u_mean))
        self.U_rho2.assign(tf.where(self.U_mask==0, -1e5*tf.ones_like(u_rho2), u_rho2))
        
        # Bias vector
        b_mean = self.b_mu
        b_rho2 = self.b_rho2
        # Reset bias
        self.b_mu.assign(tf.where(self.b_mask==0, tf.zeros_like(b_mean), b_mean))
        self.b_rho2.assign(tf.where(self.b_mask==0, -1e5*tf.ones_like(b_rho2), b_rho2))