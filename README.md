# AI Assistant – Converting Sign Language into Text

This project is a deep learning–based system that recognizes American Sign Language (ASL) alphabets and converts them into text using a Convolutional Neural Network (CNN).

The model is trained on an ASL alphabet image dataset and supports real-time sign recognition using a webcam. It aims to empower the deaf and hard-of-hearing community by improving accessibility and reducing dependency on human interpreters.

## Features
- ASL alphabet recognition (A–Z)
- CNN-based image classification
- Real-time webcam detection using OpenCV
- Image-based prediction support
- High accuracy (~98% validation accuracy)

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

## Dataset Setup
Download an ASL alphabet image dataset and place it inside the project directory.

Required folder structure:
SIGN/
├── asl_dataset/
│ ├── a/
│ ├── b/
│ ├── c/
│ ├── ...
│ └── z/

## How to Run
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies using `pip install -r requirements.txt`
4. Download the dataset and place it in the project directory
5. Run `data_preparation.py` and `train.py`
6. Run `recognize.py` for real-time detection



