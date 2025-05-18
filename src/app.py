"""
Main application for Turkish Sign Language Interpreter.

This module contains the Streamlit application for real-time Turkish Sign Language interpretation.
"""

import os
import time
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

from src.models.model_loader import load_tsl_model
from src.utils.data_processing import extract_keypoints, normalize_sequence
from src.config.constants import MAX_SEQUENCE_LENGTH, PREDICTION_THRESHOLD, CLASS_MAPPING


# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def main():
    """Main application function for the Turkish Sign Language Interpreter."""
    # Set page config
    st.set_page_config(page_title="Turkish Sign Language Interpreter", layout="wide")
    
    st.title("Turkish Sign Language Interpreter")
    
    # Add model selection in sidebar
    st.sidebar.header("Settings")
    model_option = st.sidebar.selectbox(
        "Model to use:",
        ["Standard Model", "Advanced Model"],
        index=0
    )
    
    # Determine model path based on selection
    model_path = 'models/tsl_model.keras'
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file {model_path} not found!")
        return
    
    # Load model
    try:
        model = load_tsl_model(model_path)
        st.sidebar.success(f"Model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # User configurable settings
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)
    confidence_threshold = st.sidebar.slider("Prediction Confidence Threshold", 0.1, 0.9, PREDICTION_THRESHOLD)
    
    # Display options
    show_keypoints = st.sidebar.checkbox("Show landmarks", value=True)
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    # Camera column
    with col1:
        st.header("Camera Feed")
        cam_placeholder = st.empty()
        
        # Camera button
        start_button = st.button("Start Camera")
    
    # Prediction column
    with col2:
        st.header("Recognition")
        prediction_text = st.empty()
        confidence_bar = st.empty()
        
        # Buffer display
        st.header("Sequence Buffer")
        buffer_progress = st.progress(0)
        buffer_counter = st.empty()
        
        # Add debug info
        st.subheader("Debug Info")
        debug_text = st.empty()
    
    # Run camera if button pressed
    if start_button:
        frame_buffer = []
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # Initialize holistic model
        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        ) as holistic:
            
            # Status variables
            frame_count = 0
            current_prediction = "Waiting for sign..."
            current_confidence = 0.0
            processing = False
            
            try:
                while True:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera error")
                        break
                    
                    # Skip frames for performance
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                    
                    # Flip and convert
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    results = holistic.process(rgb_frame)
                    
                    # Draw landmarks if enabled
                    if show_keypoints:
                        annotated_frame = rgb_frame.copy()
                        
                        # Draw landmarks
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    else:
                        annotated_frame = rgb_frame
                    
                    # Display frame
                    cam_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    
                    # Only add frames with meaningful data
                    if np.any(keypoints != 0):
                        frame_buffer.append(keypoints)
                        if len(frame_buffer) > MAX_SEQUENCE_LENGTH:
                            frame_buffer.pop(0)
                    
                    # Update buffer status
                    buffer_ratio = len(frame_buffer) / MAX_SEQUENCE_LENGTH
                    buffer_progress.progress(buffer_ratio)
                    buffer_counter.text(f"Frames: {len(frame_buffer)}/{MAX_SEQUENCE_LENGTH}")
                    
                    # Make prediction when buffer is full
                    if len(frame_buffer) == MAX_SEQUENCE_LENGTH and not processing:
                        processing = True
                        
                        # Prepare sequence - normalize using the same method as in training
                        sequence = np.array(frame_buffer)
                        sequence_norm = normalize_sequence(sequence)
                        sequence_batch = np.expand_dims(sequence_norm, axis=0)
                        
                        # Predict
                        prediction = model.predict(sequence_batch, verbose=0)[0]
                        class_idx = np.argmax(prediction)
                        confidence = prediction[class_idx]
                        
                        # Update prediction if confidence is high enough
                        if confidence >= confidence_threshold:
                            current_prediction = CLASS_MAPPING.get(class_idx, "Unknown")
                            current_confidence = confidence
                        
                        # Reset for next prediction
                        frame_buffer = frame_buffer[-15:]
                        processing = False
                    
                    # Display prediction
                    prediction_style = "color:green;" if current_confidence >= confidence_threshold else "color:gray;"
                    prediction_text.markdown(f"<h2 style='text-align:center;{prediction_style}'>{current_prediction}</h2>", 
                                           unsafe_allow_html=True)
                    confidence_bar.progress(float(current_confidence))
                    
                    # Sleep to reduce CPU usage
                    time.sleep(0.01)
                    
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Release resources
                cap.release()
    
    else:
        # Display placeholder when camera is not running
        cam_placeholder.image("https://via.placeholder.com/640x480.png?text=Camera+Off", use_container_width=True)
        
        # Instructions
        st.markdown("""
        ## Instructions
        1. Click "Start Camera" to begin
        2. Position yourself in the camera view
        3. Perform Turkish Sign Language signs
        4. The model will attempt to recognize your signs
        
        ### Tips for better recognition:
        - Ensure good lighting
        - Position your hands clearly in view
        - Try to match the speed of signs from the training data
        - Adjust the confidence threshold if needed
        """)


if __name__ == "__main__":
    main()
