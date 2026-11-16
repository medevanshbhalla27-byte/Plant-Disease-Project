"""Step 3: Normalize image pixel values and optimize the input pipeline."""
import tensorflow as tf
from step2 import train_ds as raw_train_ds, val_ds as raw_val_ds, class_names

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

train_ds = raw_train_ds.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
)
val_ds = raw_val_ds.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Prefetch overlaps data preparation with model execution.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
