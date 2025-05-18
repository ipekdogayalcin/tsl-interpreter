# Turkish Sign Language Interpreter

A real-time Turkish Sign Language (TSL) interpretation system using computer vision and deep learning.

## Overview

This project provides a real-time Turkish Sign Language interpretation system that uses a webcam to capture sign language gestures and translates them into text. The system utilizes MediaPipe for pose and hand landmark detection and a trained deep learning model for sign classification.

## Features

- Real-time sign language detection and interpretation
- Support for 226 Turkish Sign Language signs
- Interactive Streamlit web interface
- Adjustable confidence threshold for predictions
- Visual feedback with landmark visualization

## Repository Structure

```
tsl_interpreter_project/
├── models/                  # Trained model files
│   └── tsl_model.keras      # Main TSL interpreter model
├── src/                     # Source code
│   ├── config/              # Configuration modules
│   │   └── constants.py     # Constants and class mappings
│   ├── models/              # Model-related code
│   │   ├── layers.py        # Custom model layers
│   │   └── model_loader.py  # Model loading utilities
│   ├── utils/               # Utility functions
│   │   └── data_processing.py  # Data processing utilities
│   └── app.py               # Main Streamlit application
├── .gitignore               # Git ignore file
├── LICENSE                  # License file
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/zs07/tsl-interpreter.git
   cd tsl-interpreter
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. In the web interface:
   - Click "Start Camera" to begin capturing video
   - Perform Turkish Sign Language signs in front of the camera
   - Adjust settings in the sidebar as needed
   - View real-time predictions in the right panel

## Model Information

The TSL interpreter uses a deep learning model trained on a dataset of Turkish Sign Language signs. The model architecture includes:

- LSTM layers for temporal sequence processing
- Custom Attention mechanism for focusing on the most relevant parts of the sign sequence
- Dense layers for classification

The model takes as input a sequence of pose and hand landmarks extracted using MediaPipe and outputs probabilities for each of the 226 supported signs.

For detailed training code and experiments, see the [Kaggle training notebook](https://www.kaggle.com/code/zzzz07/signspeakmodel-1).

## Supported Signs (Sample)

| ID | Sign (Turkish) | Meaning (English)     |
|----|----------------|-----------------------|
| 0  | abla           | sister                |
| 1  | acele          | hurry                 |
| 2  | acikmak        | hungry                |
| 14 | anne           | mother                |
| 20 | baba           | father                |
| 35 | calismak       | work                  |
| 50 | devlet         | government            |
| 65 | evet           | yes                   |
| 86 | hayir          | no                    |
| 114| kitap          | book                  |

*For the complete list of 226 signs, please refer to `src/config/constants.py`.*


## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- Streamlit
- NumPy

## License

[MIT License](LICENSE)

## Acknowledgments
- [AUTSL Dataset](https://cvml.ankara.edu.tr/datasets/)
- MediaPipe for the pose and hand landmark detection
- TensorFlow and Keras for the deep learning framework
- Streamlit for the interactive web interface
