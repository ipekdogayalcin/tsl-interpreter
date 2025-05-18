"""
Custom layers for Turkish Sign Language Interpreter models.

This module contains custom Keras layers used in the TSL Interpreter model architecture.
"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class Attention(Layer):
    """
    Custom attention mechanism layer for sequence models.
    
    This layer applies an attention mechanism to input sequences, allowing the model
    to focus on the most relevant parts of the sequence for prediction.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Attention layer."""
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], 1),
            initializer='uniform',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        """
        Apply the attention mechanism to the input tensor.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Tensor after applying attention mechanism
        """
        # Apply attention mechanism
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor
            
        Returns:
            Output shape tuple
        """
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super(Attention, self).get_config()
        return config
