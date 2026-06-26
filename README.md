
# Skin Cancer Detection Using CNN and ResNet50

A deep learning project for classifying skin lesions as **Benign** or **Malignant** using Convolutional Neural Networks (CNN) and ResNet50.

## Features
- Image preprocessing
- CNN baseline model
- ResNet50 transfer learning model
- Training, validation and testing pipeline
- Model saving and inference script

## Dataset
ISIC Skin Cancer Dataset (224x224 RGB images).

## Tech Stack
Python, TensorFlow, Keras, NumPy, OpenCV, Scikit-learn, Matplotlib.

## Folder Structure
```
Skin-Cancer-Detection-Using-CNN-and-ResNet50/
│── train.py
│── predict.py
│── requirements.txt
│── README.md
│── dataset/
│   ├── train/
│   └── test/
```

## Run

```bash
pip install -r requirements.txt
python train.py
python predict.py sample.jpg
```

## Results
The original academic project compared a custom CNN with ResNet50, where the report observed approximately **69.85%** CNN accuracy and **82%** ResNet50 accuracy on the evaluated dataset.

## Future Improvements
- EfficientNet
- Vision Transformers
- Explainable AI (Grad-CAM)
- FastAPI deployment
- Docker support
