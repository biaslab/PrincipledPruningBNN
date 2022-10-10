# Imports
from keras import backend as K
from keras import initializers, activations
import tensorflow as tf



# Dense layer
class DenseBBB(tf.keras.layers.Layer):
    """
    Variational fully connected layer (dense), following Bayes-by-Backprop (BBB).
    
    It takes the number of units as its input, all other inputs are optional.
    """
    
    def __init__(self, 
                 units,                     # number of output features
                 activation =  None,        # activation function
                 reparam    = 'local',      # which reparameterization
                 prior_var  =  1.,          # prior variance of parameters
                 std_dev    =  0.,          # standard deviation of initializer
                 init       = 'prior',      # manner in which params are initialized
                 seed       =  None,        # seed for (param) initialization
                 **kwargs):
        
        # Copy inputs ...
        self.units      = units
        self.activation = activations.get(activation)
        self.reparam    = reparam
        self.prior_var  = prior_var
        self.std_dev    = std_dev
        self.init       = init
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
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(self.prior_var)) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'he':       # 'he' uses mean and variance from HeNormal
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(2. / input_shape[1])) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'glorot':   # 'glorot' uses mean and variance from GlorotNormal
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(2. / (input_shape[1] + self.units))) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'paper':    # 'paper' follows Haussmann et al. (2019)
            self.init_mu  = initializers.HeNormal()
            self.init_rho = initializers.normal(mean=-4.5, stddev=1e-3)
        elif self.init == 'tf':       # 'tf' follows the TensorFlow implementation
            self.init_mu   = initializers.normal(mean=0., stddev=0.1)
            self.init_rho2 = initializers.normal(mean=-6., stddev=0.1)
        
        
        # Weight matrix 'W', also called kernel
        self.kernel_mu  = self.add_weight(name='kernel_mu', shape=(input_shape[1], self.units),
                                          initializer=self.init_mu, trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=(input_shape[1], self.units),
                                          initializer=self.init_rho, trainable=True)
        
        # Bias vector 'b'
        self.bias_mu    = self.add_weight(name='bias_mu', shape=(self.units,),
                                          initializer=self.init_mu, trainable=True)
        self.bias_rho   = self.add_weight(name='bias_rho', shape=(self.units,),
                                          initializer=self.init_rho, trainable=True)
        
        # Add KL-divergence loss
        self.add_loss(lambda: self.KL())
        
        # Create masks for pruning
        self.kernel_mask = tf.ones_like(self.kernel_mu)
        self.bias_mask   = tf.ones_like(self.bias_mu)
        
        # Super build function
        super().build(input_shape)
        
    
    # Standard function to compute output on forward pass
    def call(self, inputs, **kwargs):
        
        # For local reparameterization
        if self.reparam == 'local':
            # Get weight and bias variances
            kernel_sigma = tf.math.softplus(self.kernel_rho)
            bias_sigma   = tf.math.softplus(self.bias_rho)
            # Get output mean and variance
            out_mu    = K.dot(inputs, self.kernel_mu) + self.bias_mu
            out_sigma = K.dot(K.square(inputs), K.square(kernel_sigma)) + K.square(bias_sigma)
            # Sample from output
            y = out_mu + K.sqrt(out_sigma) * tf.random.normal(tf.shape(out_mu))
        
        # For global reparameterization
        else:
            # Sample weight matrix 'W'
            kernel_sigma = tf.math.softplus(self.kernel_rho)
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            # Sample bias vector 'b'
            bias_sigma = tf.math.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            # Compute output sample
            y = K.dot(inputs, kernel) + bias

        # Return layer output
        return self.activation(y)
    
    
    # Custom function to compute KL-divergence loss of layer
    def KL(self):
        
        # Kernel
        w_mean = self.kernel_mu
        w_var  = K.square(K.softplus(self.kernel_rho))
        w_vals = (w_var + K.square(w_mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(w_var)
        KL_w   = 0.5 * K.sum(tf.boolean_mask(w_vals, tf.math.is_finite(w_vals)))
        
        # Bias
        b_mean = self.bias_mu
        b_var  = K.square(K.softplus(self.bias_rho))
        b_vals = (b_var + K.square(b_mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(b_var)
        KL_b   = 0.5 * K.sum(tf.boolean_mask(b_vals, tf.math.is_finite(b_vals)))
        
        # Return sum of kernel and bias
        return KL_w + KL_b
    
    
    # Custom function for compression based on BMR
    def compress(self, red_var=1e-16):
        
        # Kernel matrix
        w_mean = self.kernel_mu
        w_rho  = self.kernel_rho
        w_var  = K.square(K.softplus(w_rho))
        # Compute BMR values
        BMR_w = self.BMR(w_mean, w_var, red_var)
        # Compress parameters with dVFE <= 0
        self.kernel_mu.assign(tf.where(BMR_w<=0, tf.zeros_like(w_mean), w_mean))
        self.kernel_rho.assign(tf.where(BMR_w<=0, -1e5*tf.ones_like(w_rho), w_rho))
        # Update kernel mask
        self.kernel_mask = tf.where(BMR_w<=0, tf.zeros_like(w_mean), self.kernel_mask)
        
        # Bias vector
        b_mean = self.bias_mu
        b_rho  = self.bias_rho
        b_var  = K.square(K.softplus(b_rho))
        # Compute BMR values
        BMR_b = self.BMR(b_mean, b_var, red_var)
        # Compress parameters with dVFE <= 0
        self.bias_mu.assign(tf.where(BMR_b<=0, tf.zeros_like(b_mean), b_mean))
        self.bias_rho.assign(tf.where(BMR_b<=0, -1e5*tf.ones_like(b_rho), b_rho))
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
        w_rho  = self.kernel_rho
        # Reset kernel
        self.kernel_mu.assign(tf.where(self.kernel_mask==0, tf.zeros_like(w_mean), w_mean))
        self.kernel_rho.assign(tf.where(self.kernel_mask==0, -1e5*tf.ones_like(w_rho), w_rho))
        
        # Bias vector
        b_mean = self.bias_mu
        b_rho  = self.bias_rho
        # Reset bias
        self.bias_mu.assign(tf.where(self.bias_mask==0, tf.zeros_like(b_mean), b_mean))
        self.bias_rho.assign(tf.where(self.bias_mask==0, -1e5*tf.ones_like(b_rho), b_rho))



# Custom Gamma layer
class GammaBBB(tf.keras.layers.Layer):
    """
    Dummy layer for adding an alpha and beta parameter of a Gamma distribution to a BNN. 
    
    Allows for joint optimization of posterior precision parameter(s).
    """
    
    def __init__(self,
                 units = 1,                 # number of output features
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
        
        # Extend alpha and beta to match inputs size
        alpha = K.exp(self.log_alpha) * tf.ones_like(inputs)
        beta  = K.exp(self.log_beta) * tf.ones_like(inputs)
        
        # Return inputs incl. alpha and beta
        return tf.stack([inputs, alpha, beta], axis=-1)
    
    
    # Custom function for KL-divergence
    def KL(self):
        
        # Get alpha and beta
        alpha = K.exp(self.log_alpha)
        beta  = K.exp(self.log_beta)
        
        # Return KL-divergence
        return K.sum((alpha - 1) * tf.math.digamma(alpha) - tf.math.lgamma(alpha) + tf.math.log(beta) + alpha * ((1 - beta) / beta))



# GRU cell (i.e. layer)
class GRUCellBBB(tf.keras.layers.Layer):
    """
    Variational Gated Recurrent Unit (GRU), following Bayes-by-Backprop (BBB).
    
    It takes the number of units as its input, all other inputs are optional.
    """
    
    def __init__(self,
                 units,                     # number of output features
                 reparam    = 'local',      # which reparameterization
                 prior_var  =  1.,          # prior variance of parameters
                 std_dev    =  0.,          # standard deviation of initializer
                 init       = 'prior',      # manner in which params are initialized
                 seed       =  None,        # seed for (param) initialization
                 **kwargs):
        
        # Copy inputs ...
        self.units      = 3*units
        self.state_size = units
        self.reparam    = reparam
        self.prior_var  = prior_var
        self.std_dev    = std_dev
        self.init       = init
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
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(self.prior_var)) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'he':       # 'he' uses mean and variance from HeNormal
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(2. / input_shape[1])) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'glorot':   # 'glorot' uses mean and variance from GlorotNormal
            self.init_mu  = initializers.normal(mean=0., stddev=self.std_dev)
            self.init_rho = initializers.normal(mean=K.log(K.exp(tf.math.sqrt(2. / (input_shape[1] + self.units))) - 1.), 
                                                stddev=self.std_dev)
        elif self.init == 'paper':    # 'paper' follows Haussmann et al. (2019)
            self.init_mu  = initializers.HeNormal()
            self.init_rho = initializers.normal(mean=-4.5, stddev=1e-3)
        elif self.init == 'tf':       # 'tf' follows the TensorFlow implementation
            self.init_mu   = initializers.normal(mean=0., stddev=0.1)
            self.init_rho2 = initializers.normal(mean=-6., stddev=0.1)
        
        # Kernel matrix
        self.W_mu  = self.add_weight(name='W_mu', shape=(input_shape[1], self.units),
                                     initializer=self.init_mu, trainable=True)
        self.W_rho = self.add_weight(name='W_rho', shape=(input_shape[1], self.units),
                                     initializer=self.init_rho, trainable=True)
        # Hidden matrix
        self.U_mu  = self.add_weight(name='U_mu', shape=(self.state_size, self.units),
                                     initializer=self.init_mu, trainable=True)
        self.U_rho = self.add_weight(name='U_rho', shape=(self.state_size, self.units),
                                     initializer=self.init_rho, trainable=True)
        # Bias vector
        self.b_mu  = self.add_weight(name='b_mu', shape=(self.units,),
                                     initializer=self.init_mu, trainable=True)
        self.b_rho = self.add_weight(name='b_rho', shape=(self.units,),
                                     initializer=self.init_rho, trainable=True)
        
        # Sampling noise
        if self.reparam == 'local':
            self.r_eps     = tf.Variable(tf.zeros(int(self.units/3)), trainable=False)
            self.u_eps     = tf.Variable(tf.zeros(int(self.units/3)), trainable=False)
            self.h_pre_eps = tf.Variable(tf.zeros(int(self.units/3)), trainable=False)
        else:
            self.W_eps = tf.Variable(tf.zeros_like(self.W_rho), trainable=False)
            self.U_eps = tf.Variable(tf.zeros_like(self.U_rho), trainable=False)
            self.b_eps = tf.Variable(tf.zeros_like(self.b_rho), trainable=False)
        
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
        
        # Get state value
        h_min1 = states[0]
        
        # Sample noise for first time step
        if K.sum(h_min1) == 0:
            self.sample_noise()
        
        # For local reparameterization
        if self.reparam == 'local':
            # Split means ...
            W_r_mu, W_u_mu, W_h_mu = tf.split(self.W_mu, 3, axis=1)
            U_r_mu, U_u_mu, U_h_mu = tf.split(self.U_mu, 3, axis=1)
            b_r_mu, b_u_mu, b_h_mu = tf.split(self.b_mu, 3, axis=0)
            # ... and variances
            W_r_sig, W_u_sig, W_h_sig = tf.split(K.softplus(self.W_rho), 3, axis=1)
            U_r_sig, U_u_sig, U_h_sig = tf.split(K.softplus(self.U_rho), 3, axis=1)
            b_r_sig, b_u_sig, b_h_sig = tf.split(K.softplus(self.b_rho), 3, axis=0)
            # Reset gate
            r_mu  = K.dot(inputs, W_r_mu) + K.dot(h_min1, U_r_mu) + b_r_mu
            r_sig = K.dot(K.square(inputs), K.square(W_r_sig)) + K.dot(K.square(h_min1), K.square(U_r_sig)) + b_r_sig
            r     = tf.math.sigmoid(r_mu + r_sig * self.r_eps)
            # Update gate
            u_mu  = K.dot(inputs, W_u_mu) + K.dot(h_min1, U_u_mu) + b_u_mu
            u_sig = K.dot(K.square(inputs), K.square(W_u_sig)) + K.dot(K.square(h_min1), K.square(U_u_sig)) + b_u_sig
            u     = tf.math.sigmoid(u_mu + u_sig * self.u_eps)
            # Hidden unit pre
            h_pre_mu  = K.dot(inputs, W_h_mu) + K.dot(h_min1, U_h_mu) + b_h_mu
            h_pre_sig = K.dot(K.square(inputs), K.square(W_h_sig)) + K.dot(K.square(h_min1), K.square(U_h_sig)) + b_h_sig
            h_pre     = tf.math.tanh(h_pre_mu + h_pre_sig * self.h_pre_eps)
            # Hidden unit final
            h = u * h_min1 + (1. - u) * h_pre
        
        # For global reparameterization:
        else:
            # Sample and split parameters
            W_r, W_u, W_h = tf.split(self.W_mu + K.softplus(self.W_rho) * self.W_eps, 3, axis=1)
            U_r, U_u, U_h = tf.split(self.U_mu + K.softplus(self.U_rho) * self.U_eps, 3, axis=1)
            b_r, b_u, b_h = tf.split(self.b_mu + K.softplus(self.b_rho) * self.b_eps, 3, axis=0)
            # Reset gate
            r = tf.math.sigmoid(K.dot(inputs, W_r) + K.dot(h_min1, U_r) + b_r)
            # Update gate
            u = tf.math.sigmoid(K.dot(inputs, W_u) + K.dot(h_min1, U_u) + b_u)
            # Hidden unit pre
            h_pre = tf.math.tanh(K.dot(inputs, W_h) + K.dot(r * h_min1, U_h) + b_h)
            # Hidden unit final
            h = u * h_min1 + (1. - u) * h_pre
        
        # Return cell output and state
        return h, [h]
    
    
    # Custom function for sampling noise matrices and vectors
    def sample_noise(self):
        
        # Local reparameterization
        if self.reparam == 'local':
            self.r_eps.assign(tf.random.normal(tf.shape(self.r_eps)))
            self.u_eps.assign(tf.random.normal(tf.shape(self.u_eps)))
            self.h_pre_eps.assign(tf.random.normal(tf.shape(self.h_pre_eps)))
        
        # Global reparameterization
        else:
            self.W_eps.assign(tf.random.normal(tf.shape(self.W_eps)))
            self.U_eps.assign(tf.random.normal(tf.shape(self.U_eps)))
            self.b_eps.assign(tf.random.normal(tf.shape(self.b_eps)))
    
    
    # Custom function to compute KL-divergence values given mean and std.dev.
    def kl_value(self, mean, std):
        
        # Get variance
        var = K.square(std)
        # KL-divergence values
        KL = (var + K.square(mean)) / self.prior_var - 1. + K.log(self.prior_var) - K.log(var)
        
        # Return filtered values
        return 0.5 * K.sum(tf.boolean_mask(KL, tf.math.is_finite(KL)))
    
    
    # Custom function to compute total KL-divergence loss of layer
    def KL(self):
        
        # Get all variances
        W_sig, U_sig, b_sig = K.softplus(self.W_rho), K.softplus(self.U_rho), K.softplus(self.b_rho)
        
        # Return values
        return self.kl_value(self.W_mu, W_sig) + self.kl_value(self.U_mu, U_sig) + self.kl_value(self.b_mu, b_sig)
    
    
    # Custom function for compression based on BMR
    def compress(self, red_var=1e-16):
        
        # Kernel matrix
        w_mean = self.W_mu
        w_rho  = self.W_rho
        w_var  = K.square(K.softplus(w_rho))
        # Compute BMR values
        BMR_w = self.BMR(w_mean, w_var, red_var)
        # Compress parameters with dVFE <= 0
        self.W_mu.assign(tf.where(BMR_w<=0, tf.zeros_like(w_mean), w_mean))
        self.W_rho.assign(tf.where(BMR_w<=0, -1e5*tf.ones_like(w_rho), w_rho))
        # Update kernel mask
        self.W_mask = tf.where(BMR_w<=0, tf.zeros_like(w_mean), self.W_mask)
        
        # Hidden matrix
        u_mean = self.U_mu
        u_rho  = self.U_rho
        u_var  = K.square(K.softplus(u_rho))
        # Compute BMR values
        BMR_u = self.BMR(u_mean, u_var, red_var)
        # Compress parameters with dVFE <= 0
        self.U_mu.assign(tf.where(BMR_u<=0, tf.zeros_like(u_mean), u_mean))
        self.U_rho.assign(tf.where(BMR_u<=0, -1e5*tf.ones_like(u_rho), u_rho))
        # Update hidden mask
        self.U_mask = tf.where(BMR_u<=0, tf.zeros_like(u_mean), self.U_mask)
        
        # Bias
        b_mean = self.b_mu
        b_rho  = self.b_rho
        b_var  = K.square(K.softplus(b_rho))
        # Compute BMR values
        BMR_b = self.BMR(b_mean, b_var, red_var)
        # Compress parameters with dVFE <= 0
        self.b_mu.assign(tf.where(BMR_b<=0, tf.zeros_like(b_mean), b_mean))
        self.b_rho.assign(tf.where(BMR_b<=0, -1e5*tf.ones_like(b_rho), b_rho))
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
        w_rho  = self.W_rho
        # Reset kernel
        self.W_mu.assign(tf.where(self.W_mask==0, tf.zeros_like(w_mean), w_mean))
        self.W_rho.assign(tf.where(self.W_mask==0, -1e5*tf.ones_like(w_rho), w_rho))
        
        # Hidden matrix
        u_mean = self.U_mu
        u_rho  = self.U_rho
        # Reset hidden
        self.U_mu.assign(tf.where(self.U_mask==0, tf.zeros_like(u_mean), u_mean))
        self.U_rho.assign(tf.where(self.U_mask==0, -1e5*tf.ones_like(u_rho), u_rho))
        
        # Bias vector
        b_mean = self.b_mu
        b_rho  = self.b_rho
        # Reset bias
        self.b_mu.assign(tf.where(self.b_mask==0, tf.zeros_like(b_mean), b_mean))
        self.b_rho.assign(tf.where(self.b_mask==0, -1e5*tf.ones_like(b_rho), b_rho))