"""
Package initialization file for the TSL Interpreter project.

This file ensures that the package is properly initialized and modules can be imported.
"""

# Import key modules for easier access
from src.models.model_loader import load_tsl_model
from src.utils.data_processing import extract_keypoints, normalize_sequence
