# Plant Disease Detection using CNN

A deep-learning image classification project that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify plant leaf images into the categories represented in the PlantVillage dataset.

> **Note:** This is an image-classification project. It predicts a class for a supplied leaf image; it is not a field-tested diagnostic system.

## Project workflow

1. **Dataset exploration** — inspect class folders, image counts, and a sample image.
2. **Dataset loading** — load images from class-specific folders and create training/validation splits.
3. **Normalization** — scale pixel values from `[0, 255]` to `[0, 1]` and prefetch batches.
4. **CNN architecture** — build a CNN with convolution, max-pooling, global average pooling, dense, and dropout layers.
5. **Training and saving** — apply data augmentation, train the model, save the best checkpoint, and save the final model.

## Model architecture

- Input: `224 × 224 × 3`
- Data augmentation: random horizontal flip, rotation, and zoom
- Convolution blocks: 32, 64, and 128 filters
- Max pooling after each convolution
- Global Average Pooling
- Dense layer: 128 units with ReLU
- Dropout: `0.3`
- Output: softmax with one unit per dataset class

The model uses Adam, sparse categorical cross-entropy, and accuracy as its training metric. Training includes model checkpointing, early stopping, and learning-rate reduction on a validation-loss plateau.

## Project structure

```text
Plant-Disease-Detection/
├── config.py
├── step1.py
├── step2.py
├── step3.py
├── step4.py
├── step5.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/                 # created during training; ignored by Git
    ├── best_model.keras
    ├── plant_model.keras
    └── class_names.json
```

## Dataset setup

Download the PlantVillage dataset separately and arrange it in class-specific folders:

```text
PlantVillage/
├── Tomato___healthy/
│   ├── image_001.jpg
│   └── ...
├── Tomato___Early_blight/
│   ├── image_001.jpg
│   └── ...
└── ...
```

The folder name is used as the class label. The dataset itself is not included in this repository.

By default, `config.py` uses:

```python
r"C:\Users\LENOVO\Downloads\PlantVillage"
```

Change `DATASET_DIR` in `config.py` to your dataset location. Alternatively, set the `PLANTVILLAGE_DIR` environment variable.

**PowerShell example:**

```powershell
$env:PLANTVILLAGE_DIR = "C:\path\to\PlantVillage"
```

## Installation

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a TensorFlow build compatible with your Python version and hardware if the default installation is not suitable for your system.

## Run the project

Run these commands from the project root, in order:

```powershell
python step1.py
python step2.py
python step4.py
python step5.py
```

`step3.py` is imported by `step5.py` and applies normalization to the training and validation datasets. `step2.py` is also imported by later steps to keep class labels consistent.

After training, the `outputs/` directory contains:

- `best_model.keras` — checkpoint with the highest validation accuracy.
- `plant_model.keras` — final model saved after training/early stopping.
- `class_names.json` — ordered class labels required for prediction.

## Predict an image

After training, run:

```powershell
python predict.py "C:\path\to\leaf_image.jpg"
```

The script prints the predicted class and the model's softmax score for that class.

## Important evaluation note

The validation split is used during model development and checkpoint selection. It is not an independent test set. For a reliable estimate of performance on unseen images, keep a separate test dataset that is not used during training or model selection.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- CNN
- Image preprocessing and data augmentation

## Learning outcomes

- Image dataset exploration and loading
- Image resizing and pixel normalization
- Training/validation dataset splitting
- CNN model construction
- Data augmentation
- Model checkpointing and training callbacks
- Saving a trained model and running single-image predictions
