"""
Model loading utilities for Turkish Sign Language Interpreter.

This module provides functions for loading and initializing the TSL model.
"""

import streamlit as st
from tensorflow.keras.models import load_model

from src.models.layers import Attention


@st.cache_resource
def load_tsl_model(model_path='models/tsl_model.keras'):
    """
    Load the Turkish Sign Language model with custom layers.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded Keras model
    """
    try:
        # Load model with custom_objects
        model = load_model(
            model_path, 
            custom_objects={'Attention': Attention},
            compile=False
        )
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
