"""
Utility functions for data processing in the Turkish Sign Language Interpreter.

This module contains functions for keypoint extraction, sequence normalization,
and other data processing operations used in the TSL Interpreter.
"""

import numpy as np
import mediapipe as mp


# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results.
    
    Args:
        results: MediaPipe holistic model results
        
    Returns:
        numpy.ndarray: Flattened array of pose and hand keypoints
    """
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    
    # Extract hand landmarks
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # Concatenate
    return np.concatenate([pose, left_hand, right_hand])


def normalize_sequence(sequence):
    """
    Normalize a sequence of keypoints.
    
    Takes a single sequence and normalizes it by subtracting the mean and
    dividing by the standard deviation of non-zero values.
    
    Args:
        sequence: numpy.ndarray of shape (frames, features)
        
    Returns:
        numpy.ndarray: Normalized sequence
    """
    # Take a single sequence and normalize it
    norm_seq = np.zeros_like(sequence)
    
    # Find non-zero values
    mask = sequence != 0
    if np.any(mask):
        # Get stats only from non-zero values
        mean = np.mean(sequence[mask])
        std = np.std(sequence[mask])
        # Normalize with small epsilon to avoid division by zero
        if std > 1e-6:
            norm_seq[mask] = (sequence[mask] - mean) / std
        else:
            norm_seq[mask] = sequence[mask] - mean
    
    return norm_seq
