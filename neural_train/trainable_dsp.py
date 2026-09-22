import tensorflow as tf
import numpy as np

# Purely functional LMS filter (uses no tf.Variable)
def lms_filter(x, d, weights, mu, filter_size):
    """
    Purely functional LMS filter
    
    Args:
        x: input signal [batch_size, seq_len]
        d: reference (desired) signal [batch_size, seq_len]
        weights: filter coefficients [filter_size]
        mu: step size parameter
        filter_size: filter order
    """
    batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
    
    # Initialize output and error
    y = tf.zeros_like(x)
    e = tf.zeros_like(x)
    
    # Record the weights at every step (for gradient computation)
    weights_history = [weights]
    
    # LMS iterations (tf.while_loop instead of a Python loop)
    def loop_body(i, y, e, current_weights, weights_history):
        # Extract the current input frame
        x_frame = x[:, i-filter_size+1:i+1]  # [batch, filter_size]
        
        # Compute the filter output
        y_i = tf.reduce_sum(current_weights * x_frame, axis=1)  # [batch]
        
        # Compute the error
        e_i = d[:, i] - y_i  # [batch]
        
        # Update the output and error tensors
        y = tf.tensor_scatter_nd_update(y, [[0, i]], [y_i])
        e = tf.tensor_scatter_nd_update(e, [[0, i]], [e_i])
        
        # Update the filter coefficients (functionally, no assign)
        weight_update = mu * tf.reduce_mean(
            tf.expand_dims(e_i, axis=1) * x_frame, 
            axis=0
        )  # averaged update
        new_weights = current_weights + weight_update
        
        # Record the weight history
        weights_history = weights_history.write(i-filter_size+1, new_weights)
        
        return i+1, y, e, new_weights, weights_history
    
    # Initialize the weight-history buffer
    weights_history = tf.TensorArray(
        dtype=tf.float32, 
        size=seq_len-filter_size+1,
        element_shape=(filter_size,)
    )
    
    # Run the loop
    _, y, e, final_weights, _ = tf.while_loop(
        cond=lambda i, *_: i < seq_len,
        body=loop_body,
        loop_vars=(
            filter_size-1, 
            y, 
            e, 
            weights, 
            weights_history
        )
    )
    
    return y, e, final_weights

# Custom gradient function
@tf.custom_gradient
def trainable_lms_step(x, d, weights, mu, filter_size):
    """
    LMS step with a custom gradient
    
    Args:
        x: input signal
        d: reference (desired) signal
        weights: filter coefficients
        mu: step size parameter
        filter_size: filter order
    """
    y, e, final_weights = lms_filter(x, d, weights, mu, filter_size)
    
    def grad(dy, de):
        # Simplified gradient computation (a real application may need
        # a more elaborate backward pass)
        # Only the gradients w.r.t. weights and mu are returned here
        
        # Gradient w.r.t. weights
        weight_grad = -2.0 * tf.reduce_mean(de * x[:, -filter_size:], axis=0)
        
        # Gradient w.r.t. mu (simplified)
        mu_grad = tf.reduce_mean(de * e[:, -1:])
        
        return None, None, weight_grad, mu_grad, None  # only weights and mu carry gradients
    
    return (y, e), grad

# External parameter manager
class ParameterManager:
    def __init__(self, filter_size):
        # Initialize parameters (plain NumPy arrays)
        self.weights = np.zeros(filter_size, dtype=np.float32)
        self.mu = np.array(0.01, dtype=np.float32)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def update(self, gradients):
        """Apply gradients to update the parameters"""
        # Convert the NumPy parameters to tensors
        weights_tensor = tf.convert_to_tensor(self.weights)
        mu_tensor = tf.convert_to_tensor(self.mu)
        
        # Apply the gradients
        self.optimizer.apply_gradients(zip(
            gradients, 
            [weights_tensor, mu_tensor]
        ))
        
        # Write the updated tensors back to NumPy
        self.weights = weights_tensor.numpy()
        self.mu = mu_tensor.numpy()
        
    def get_params(self):
        """Get the current parameters"""
        return self.weights, self.mu

# Training loop
def train_lms_model(x_train, d_train, filter_size=32, epochs=10):
    # Create the parameter manager
    param_manager = ParameterManager(filter_size)
    
    for epoch in range(epochs):
        # Get the current parameters
        weights, mu = param_manager.get_params()
        
        # Forward pass
        with tf.GradientTape() as tape:
            # Convert the parameters to tensors
            weights_tensor = tf.convert_to_tensor(weights)
            mu_tensor = tf.convert_to_tensor(mu)
            
            # Run the LMS filter
            y, e, _ = trainable_lms_step(
                x_train, 
                d_train, 
                weights_tensor, 
                mu_tensor, 
                filter_size
            )
            
            # Compute the loss
            loss = tf.reduce_mean(tf.square(e))
        
        # Compute the gradients
        gradients = tape.gradient(loss, [weights_tensor, mu_tensor])
        
        # Apply the gradients to update the parameters
        param_manager.update(gradients)
        
        # Print training progress
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}, Mu: {mu:.6f}")
    
    return param_manager.get_params()