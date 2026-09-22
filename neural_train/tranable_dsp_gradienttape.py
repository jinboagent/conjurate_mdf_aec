import tensorflow as tf
import numpy as np

# Purely functional LMS filter (uses no tf.Variable)
def linear_filter(x, d, weights, mu, filter_size):
    """Run the LMS filtering and return the results, without updating the weights."""
    batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
    
    # Initialize output and error
    y = tf.zeros_like(x)
    e = tf.zeros_like(x)
    
    # LMS iterations
    for i in tf.range(filter_size-1, seq_len):
        # Extract the current input frame
        x_frame = x[:, i-filter_size+1:i+1]
        
        # Compute the filter output
        y_i = tf.reduce_sum(weights * x_frame, axis=1)
        
        # Compute the error
        e_i = d[:, i] - y_i
        
        # Update the output and error tensors
        y = tf.tensor_scatter_nd_update(y, [[0, i]], [y_i])
        e = tf.tensor_scatter_nd_update(e, [[0, i]], [e_i])
    
    return y, e

# Train the LMS filter with GradientTape
def train_lms_with_gradient_tape(x, d, filter_size=32, mu=0.01, learning_rate=0.001, epochs=10):
    """
    Train the LMS filter with GradientTape
    
    Args:
        x: input signal [batch_size, seq_len]
        d: reference (desired) signal [batch_size, seq_len]
        filter_size: filter order
        mu: LMS step size parameter
        learning_rate: optimizer learning rate
        epochs: number of training epochs
    """
    # Initialize the weights (as a plain tensor, not a tf.Variable)
    weights = tf.zeros((filter_size,), dtype=tf.float32)
    
    # Create the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Make the weights a tensor the tape can track
            tape.watch(weights)
            
            # Run the LMS filter (pure function call)
            y, e = lms_filter(x, d, weights, mu, filter_size)
            
            # Compute the loss (mean squared error)
            loss = tf.reduce_mean(tf.square(e))
        
        # Compute the gradients
        gradients = tape.gradient(loss, weights)
        
        # Apply the gradients to update the weights
        # Note: this uses the optimizer's apply_gradients, not the LMS weight update rule
        optimizer.apply_gradients([(gradients, weights)])
        
        # Print training progress
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")
    
    return weights

# Generate example data
def generate_data(batch_size=1, seq_len=1000):
    t = tf.linspace(0.0, 1.0, seq_len)
    clean_signal = tf.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    # Add random noise
    noise = tf.random.normal((batch_size, seq_len), stddev=0.5)
    noisy_signal = tf.tile(tf.expand_dims(clean_signal, 0), [batch_size, 1]) + noise
    
    return noisy_signal, clean_signal[tf.newaxis, :]  # [batch_size, seq_len]

# Run the training
x_train, d_train = generate_data()
trained_weights = train_lms_with_gradient_tape(
    x_train, 
    d_train, 
    filter_size=16, 
    mu=0.01, 
    learning_rate=0.001,
    epochs=20
)

print("Trained weights:", trained_weights.numpy())