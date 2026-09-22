import tensorflow as tf
from tensorflow.keras.layers import Layer

class WrappedLMSFilter(Layer):
    """封装原始LMS算法的可训练神经网络层"""
    
    def __init__(self, filter_size, **kwargs):
        super(WrappedLMSFilter, self).__init__(**kwargs)
        self.filter_size = filter_size
    
    def build(self, input_shape):
        # 可训练的初始滤波器系数（替代DSP中的固定初始值）
        self.initial_weights = self.add_weight(
            shape=(self.filter_size,),
            initializer='zeros',
            trainable=True,
            name='initial_filter_coefficients'
        )
        
        # 可训练的步长参数（使用softplus确保正值）
        self.mu = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.01),
            trainable=True,
            name='step_size'
        )
    
    def call(self, inputs):
        # 输入: [信号, 参考信号]
        x, d = inputs
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # 使用Python控制流执行原始LMS算法
        # 注意：使用tf.py_function包装以保留原始DSP逻辑
        def lms_algorithm(x_np, d_np, initial_w, mu_val):
            batch_size, seq_len = x_np.shape
            y_np = np.zeros_like(x_np)
            e_np = np.zeros_like(x_np)
            
            for b in range(batch_size):
                # 初始化滤波器系数（使用可训练的初始值）
                w = initial_w.copy()
                
                for i in range(self.filter_size-1, seq_len):
                    # 提取当前输入帧
                    x_frame = x_np[b, i-self.filter_size+1:i+1]
                    
                    # 计算滤波输出
                    y_np[b, i] = np.dot(w, x_frame)
                    
                    # 计算误差
                    e_np[b, i] = d_np[b, i] - y_np[b, i]
                    
                    # 更新滤波器系数（使用可训练的步长）
                    w += mu_val * e_np[b, i] * x_frame
            
            return y_np, e_np
        
        # 将NumPy实现包装为TensorFlow操作
        # 注意：这不是最有效的实现方式，但保留了原始DSP逻辑
        y, e = tf.py_function(
            lms_algorithm,
            [x, d, self.initial_weights, tf.nn.softplus(self.mu)],
            [tf.float32, tf.float32]
        )
        
        # 设置形状信息（因为tf.py_function会丢失形状信息）
        y.set_shape(x.shape)
        e.set_shape(x.shape)
        
        return y, e
    
    def compute_output_shape(self, input_shape):
        return input_shape, input_shape