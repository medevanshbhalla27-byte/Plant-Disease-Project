# 🌿 Plant Disease Detection using CNN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple)

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify plant leaf images into different disease categories using the PlantVillage dataset.

The project demonstrates an end-to-end image classification workflow, including dataset exploration, image preprocessing, normalization, data augmentation, CNN architecture development, model training, and image prediction.

---

## 📌 Project Overview

Plant diseases can affect agricultural productivity and crop quality. This project explores how Computer Vision and Deep Learning can be used to identify plant leaf disease categories from images.

A CNN model is developed using TensorFlow and Keras to learn visual patterns from leaf images and classify them into the corresponding dataset categories.

## ✨ Features

- PlantVillage dataset exploration
- Image loading and preprocessing
- Image resizing to 224 × 224 pixels
- Pixel normalization (0–1)
- Training and validation dataset splitting
- Custom CNN architecture
- Data augmentation
- Model checkpointing
- Early stopping and learning rate reduction
- Trained model saving
- Single-image prediction

## 🧠 Model Architecture

The CNN architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Conv2D | 32 filters |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters |
| MaxPooling2D | 2 × 2 |
| GlobalAveragePooling2D | Feature aggregation |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | Softmax |

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Crossentropy

**Evaluation Metric:** Accuracy

## 🔄 Data Augmentation

To introduce image variations during training and improve the model's ability to generalize, the following techniques are used:

- Random Horizontal Flip
- Random Rotation (0.1)
- Random Zoom (0.1)

## 📂 Project Structure

```text
Plant-Disease-Detection/
│
├── config.py
│
├── step1.py
├── step2.py
├── step3.py
├── step4.py
├── step5.py
│
├── predict.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
    ├── best_model.keras
    ├── plant_model.keras
    └── class_names.json
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Plant-Disease-Detection.git
```

Navigate to the project directory:

```bash
cd Plant-Disease-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the **PlantVillage Dataset**.

The dataset should be organized into class-specific folders:

```text
PlantVillage/
│
├── Tomato___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Potato___healthy/
└── ...
```

Each folder represents a separate classification category.

The dataset is not included in this repository and must be downloaded separately.

Update the dataset location in `config.py` before running the project.

## 🚀 How to Run

Execute the scripts in the following order:

**Step 1 – Dataset Exploration**

```bash
python step1.py
```

**Step 2 – Dataset Loading**

```bash
python step2.py
```

**Step 3 – Image Normalization**

This step is automatically imported during training.

```bash
python -c "import step3"
```

**Step 4 – CNN Model Architecture**

```bash
python step4.py
```

**Step 5 – Model Training**

```bash
python step5.py
```

After training, the generated models and class labels are saved inside the `outputs/` directory.

## 🔍 Image Prediction

Once the model has been trained, you can classify a new leaf image:

```bash
python predict.py "path/to/leaf_image.jpg"
```

The program returns:

- Predicted plant disease category
- Prediction confidence score

## 💾 Model Outputs

| File | Description |
|---|---|
| `best_model.keras` | Best checkpoint based on validation accuracy |
| `plant_model.keras` | Final saved model |
| `class_names.json` | Class labels used during prediction |

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Convolutional Neural Networks
- Computer Vision

## 📚 Key Learnings

Through this project, I explored:

1. Working with image classification datasets.
2. Image preprocessing and normalization.
3. Building CNN architectures from scratch.
4. Implementing data augmentation.
5. Training and validating deep learning models.
6. Saving trained models using Keras.
7. Performing predictions on unseen images.

## 🔮 Future Improvements

- Experiment with Transfer Learning (MobileNetV2, EfficientNet).
- Compare different CNN architectures.
- Add a dedicated independent test dataset.
- Build a simple web interface for image uploads.
- Improve model evaluation using precision, recall and F1-score.
- Explore deployment using Flask or Streamlit.

## ⚠️ Disclaimer

This project is an educational machine learning experiment. Predictions are based on the image categories learned from the training dataset and should not be considered a substitute for professional agricultural diagnosis.

---

## 👨‍💻 Author

Devansh

Interested in Machine Learning, Deep Learning, Artificial Intelligence and Computer Vision.

---

⭐ If you find this project interesting, consider giving the repository a star!
