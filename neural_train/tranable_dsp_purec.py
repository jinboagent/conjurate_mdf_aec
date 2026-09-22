import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def lms_filter_python(x, d, weights, mu):
    """纯Python实现的LMS滤波器"""
    x = x.numpy()  # 转换为NumPy数组
    d = d.numpy()
    weights = weights.numpy()
    
    filter_size = len(weights)
    y = np.zeros_like(x)
    e = np.zeros_like(x)
    weights_history = [weights.copy()]
    
    for i in range(filter_size-1, len(x)):
        # 提取当前输入帧
        x_frame = x[i-filter_size+1:i+1]
        
        # 计算滤波输出
        y[i] = np.sum(weights * x_frame)
        
        # 计算误差
        e[i] = d[i] - y[i]
        
        # 更新权重
        weights = weights + mu * e[i] * x_frame
        weights_history.append(weights.copy())
    
    return (
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(e, dtype=tf.float32),
        tf.convert_to_tensor(weights, dtype=tf.float32)
    )

def train_lms_with_python_filter(x, d, filter_size=32, mu=0.01, learning_rate=0.001, epochs=10):
    """使用纯Python LMS滤波器训练"""
    # 初始化权重（TensorFlow张量）
    weights = tf.zeros((filter_size,), dtype=tf.float32)
    
    # 创建优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch(weights)
            
            # 调用纯Python LMS滤波器
            y, e, _ = lms_filter_python(x, d, weights, mu)
            
            # 计算损失
            loss = tf.reduce_mean(tf.square(e))
        
        # 计算梯度
        gradients = tape.gradient(loss, weights)
        
        # 应用梯度更新
        optimizer.apply_gradients(zip([1.0], [weights]))
        
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")
    
    return weights

# 生成示例数据
x_train = tf.random.normal((1000,), dtype=tf.float32)
d_train = tf.random.normal((1000,), dtype=tf.float32)

# 训练模型
trained_weights = train_lms_with_python_filter(
    x_train, 
    d_train, 
    filter_size=16,
    mu=0.01,
    learning_rate=0.001,
    epochs=5
)

print("训练后的权重:", trained_weights.numpy())