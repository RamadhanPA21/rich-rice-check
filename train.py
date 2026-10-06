import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = "C:\Rama\PROJECT KAMPUS\RICH (Rice Check)\Rice leaf disease"

TRAIN_DIR = os.path.join(BASE_DIR, "Training data")
VAL_DIR = os.path.join(BASE_DIR, "Validation data")
TEST_DIR = os.path.join(BASE_DIR, "Testing data")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS = 20


# =========================================================
# 2. CEK FOLDER DATASET
# =========================================================

print("Checking dataset folders...")

print("Training :", TRAIN_DIR)
print("Validation :", VAL_DIR)
print("Testing :", TEST_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder tidak ditemukan: {TRAIN_DIR}")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Validation folder tidak ditemukan: {VAL_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Testing folder tidak ditemukan: {TEST_DIR}")


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nLoading testing dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =========================================================
# 4. NAMA KELAS
# =========================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClass names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nJumlah kelas:", num_classes)


# =========================================================
# 5. OPTIMASI DATA PIPELINE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# =========================================================
# 6. DATA AUGMENTATION
# =========================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")


# =========================================================
# 7. MEMBUAT MODEL
# =========================================================

print("\nCreating EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Bekukan model pretrained
base_model.trainable = False


inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# =========================================================
# 8. COMPILE MODEL
# =========================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# 9. CALLBACKS
# =========================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_rice_disease_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )
]


# =========================================================
# 10. TRAINING
# =========================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================================================
# 11. EVALUASI TESTING
# =========================================================

print("\nEvaluating model on testing dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n==============================")
print("TESTING RESULT")
print("==============================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("==============================")


# =========================================================
# 12. PREDIKSI TESTING
# =========================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# =========================================================
# 14. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Confusion Matrix - Rice Disease Classification")

plt.colorbar()

plt.xticks(
    range(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# =========================================================
# 15. GRAFIK ACCURACY
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    "accuracy_graph.png",
    dpi=300
)

plt.show()


# =========================================================
# 16. GRAFIK LOSS
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    "loss_graph.png",
    dpi=300
)

plt.show()


# =========================================================
# 17. SIMPAN MODEL
# =========================================================

model.save(
    "rice_disease_model.keras"
)

print("\nModel berhasil disimpan sebagai:")
print("rice_disease_model.keras")

print("\nTraining selesai!")