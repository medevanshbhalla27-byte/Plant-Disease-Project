"""Step 5: Train, validate, checkpoint, and save the CNN model."""
import json
import tensorflow as tf

from config import EPOCHS, OUTPUT_DIR
from step2 import class_names
from step3 import train_ds, val_ds
from step4 import build_model

def main():
    model = build_model(number_of_classes=len(class_names))
    model.summary()

    best_model_path = OUTPUT_DIR / "best_model.keras"
    final_model_path = OUTPUT_DIR / "plant_model.keras"
    class_names_path = OUTPUT_DIR / "class_names.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # EarlyStopping restores the best validation-loss weights before this save.
    model.save(final_model_path)

    with class_names_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2, ensure_ascii=False)

    print("\nTraining completed.")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Final model:     {final_model_path}")
    print(f"Class labels:    {class_names_path}")

    # Report the final model's validation metrics.
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"\nValidation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
