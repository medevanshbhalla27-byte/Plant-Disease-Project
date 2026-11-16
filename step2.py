"""Step 2: Load images and create training/validation datasets."""
import tensorflow as tf
from config import (
    DATASET_DIR, IMAGE_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED
)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}. "
        "Update config.py or set PLANTVILLAGE_DIR."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

class_names = train_ds.class_names
print(f"\nClasses ({len(class_names)}):")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")
