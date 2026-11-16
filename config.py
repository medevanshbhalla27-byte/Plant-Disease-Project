"""Project configuration.

You can override the dataset directory with the PLANTVILLAGE_DIR environment
variable, or edit DATASET_DIR below.
"""
import os
from pathlib import Path

DATASET_DIR = Path(os.environ.get(
    "PLANTVILLAGE_DIR",
    r"C:\Users\LENOVO\Downloads\PlantVillage"
))

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42
EPOCHS = 10

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
