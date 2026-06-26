# 🩺 Skin Cancer Detection Using CNN and ResNet50

A Deep Learning-based skin cancer classification system that detects **Benign** and **Malignant** skin lesions using **Convolutional Neural Networks (CNN)** and **ResNet50**. This project demonstrates an end-to-end computer vision pipeline including image preprocessing, model training, evaluation, and prediction.

---

## 📌 Overview

Skin cancer is one of the most common forms of cancer worldwide. Early diagnosis significantly improves treatment outcomes. This project uses deep learning techniques to automatically classify dermoscopic skin lesion images into two categories:

- ✅ Benign
- ✅ Malignant

The project compares a custom CNN architecture with a ResNet50-based transfer learning model to evaluate classification performance.

---

# 🚀 Features

- Binary Skin Cancer Classification
- Image Preprocessing
- CNN Model Implementation
- ResNet50 Transfer Learning
- Training & Validation Pipeline
- Model Evaluation
- Prediction on New Images
- Model Saving & Loading
- Performance Visualization

---

# 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

---

# 📂 Dataset

This project uses the **ISIC (International Skin Imaging Collaboration)** Skin Cancer Dataset.

### Kaggle Dataset

https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

### Official ISIC Website

https://www.isic-archive.com/

---

## Dataset Structure

After downloading the dataset, organize it as follows:

```text
dataset/
│
├── train/
│   ├── benign/
│   └── malignant/
│
└── test/
    ├── benign/
    └── malignant/
```

All images should be resized to:

```
224 × 224 RGB
```

---

# 📥 Dataset Download

The dataset is **not included** in this repository because of GitHub file size limitations.

Download it from Kaggle and place it inside the project folder:

```text
Skin-Cancer-Detection-Using-CNN-and-ResNet50/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📁 Project Structure

```text
Skin-Cancer-Detection-Using-CNN-and-ResNet50/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│
└── screenshots/
```

---

# 🔄 Workflow

```
Skin Images
      │
      ▼
Image Loading
      │
      ▼
Image Preprocessing
      │
      ▼
Normalization
      │
      ▼
CNN / ResNet50
      │
      ▼
Training
      │
      ▼
Validation
      │
      ▼
Prediction
      │
      ▼
Benign / Malignant
```

---

# 🧠 Model Architecture

### CNN Model

- Conv2D
- MaxPooling2D
- Dropout
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Softmax Output Layer

### ResNet50 Model

- Pretrained ResNet50 Backbone
- Transfer Learning
- Global Average Pooling
- Dense Output Layer
- Softmax Classifier

---

# 📊 Results

The project compares two different Deep Learning approaches.

| Model | Accuracy |
|--------|----------|
| CNN | ~69.85% |
| ResNet50 | ~82% |

The ResNet50 model achieved significantly better performance due to transfer learning and deeper feature extraction.

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/netal17/Skin-Cancer-Detection-Using-CNN-and-ResNet50.git
```

Move into the project

```bash
cd Skin-Cancer-Detection-Using-CNN-and-ResNet50
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

```bash
python train.py
```

---

# 🔍 Prediction

```bash
python predict.py sample.jpg
```

Example Output

```
Prediction : Malignant

Confidence : 98.12%
```

---

# 📈 Future Improvements

- EfficientNet
- Vision Transformers (ViT)
- Grad-CAM Visualization
- Explainable AI
- Hyperparameter Optimization
- Data Augmentation
- Docker Deployment
- FastAPI REST API
- Streamlit Web Application

---

# 💡 Skills Demonstrated

- Computer Vision
- Deep Learning
- CNN
- ResNet50
- Transfer Learning
- Image Classification
- TensorFlow
- Keras
- Python
- Model Training
- Model Evaluation
- Medical Image Analysis

---

# 📷 Screenshots

### Dataset Samples

_Add screenshot here_

---

### CNN Training

_Add screenshot here_

---

### ResNet50 Training

_Add screenshot here_

---

### Prediction Output

_Add screenshot here_

---

# 📚 References

- ISIC Archive – https://www.isic-archive.com/
- Kaggle Skin Cancer Dataset – https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
- TensorFlow Documentation – https://www.tensorflow.org/
- Keras Documentation – https://keras.io/

---

# 📜 License

This project is intended for educational and portfolio purposes.

---

# 👨‍💻 Author

**Netal Daga**

AI / Machine Learning Engineer

- GitHub: https://github.com/netal17
- LinkedIn: https://www.linkedin.com/in/netaldaga

---

⭐ If you found this project useful, consider giving it a star!
