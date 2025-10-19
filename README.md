# Face Recognition using ResNet18

This project demonstrates a **face recognition system** built with **ResNet18**, implemented in **PyTorch**.  
It was developed as part of a deep learning exercise in Indonesia AI training, focusing on applying transfer learning for image classification tasks involving facial images.

---

## 🧠 Project Overview

The notebook trains a ResNet18 model to recognize and classify human faces from a dataset stored on Google Drive.  
It includes data preprocessing, model training, evaluation, and visualization of performance metrics such as accuracy and confusion matrix.

---

## ⚙️ Features

- Face image classification using **ResNet18**
- **Transfer Learning** with pretrained ImageNet weights
- **Data preprocessing** and augmentation with `torchvision.transforms`
- Model training and evaluation with **PyTorch**
- Performance metrics using **scikit-learn**
- Visualization of results using **Matplotlib**

---

## 📁 Project Structure
Resnet_project_Indonesia-AI/
└── FaceRecog_Using_ResNet18.ipynb

---

## 🧩 Requirements

To run this notebook, install the following dependencies:

```
pip install torch torchvision scikit-learn matplotlib pillow pandas
```

---

## 🚀 How to Run
1. Open the notebook in Google Colab or Jupyter:

FaceRecog_Using_ResNet18.ipynb

2. Mount your Google Drive to access the dataset:

```
from google.colab import drive
drive.mount('/content/drive')
```

3. Update the dataset path inside the notebook:

```
data_path = 'drive/MyDrive/FaceRecognition/Dataset'
```

4. Run all cells sequentially to:
   - Load and preprocess images
   - Train the ResNet18 model
   - Evaluate performance on the test set

---

## 📊 Example Results
The notebook provides:
  - Accuracy, precision, recall, and F1-score reports
  - Confusion matrix visualization
  - Random sample predictions

Example:
Accuracy: 95.2%
Precision: 0.94
Recall: 0.95

---

## 🧑‍💻 Author
Hidayat Bagus Padwaduta, CV3 Takeo Kanade, Indonesia AI Deep Learning Program



