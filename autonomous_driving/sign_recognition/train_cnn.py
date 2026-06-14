import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import pandas as pd

# =========================
# SETTINGS
# =========================
IMG_SIZE    = 32
NUM_CLASSES = 43

# =========================
# PATHS
# =========================
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE, 'data', 'signs', 'gtsrb_new', 'Train.csv')
IMG_BASE  = os.path.join(BASE, 'data', 'signs', 'gtsrb_new')

print("=" * 55)
print("  CNN TRAINING - AUTONOMOUS DRIVING SYSTEM")
print("=" * 55)
print(f"  Project root : {BASE}")
print(f"  CSV path     : {CSV_PATH}")
print(f"  CSV exists   : {os.path.exists(CSV_PATH)}")
print("=" * 55)

if not os.path.exists(CSV_PATH):
    print("ERROR: Train.csv not found.")
    exit()

# =========================
# LOAD DATA FROM CSV
# =========================
def load_data(csv_path, img_base):
    df = pd.read_csv(csv_path)
    print(f"\n  Total rows : {len(df)}")

    images, labels = [], []

    for idx, row in df.iterrows():
        img_path = os.path.join(img_base, row['Path'])
        label    = int(row['ClassId'])

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

        if (idx + 1) % 5000 == 0:
            print(f"  Loaded {idx+1}/{len(df)}...")

    print(f"\n  Images loaded : {len(images)}")
    print(f"  Classes found : {len(set(labels))}")
    return np.array(images), np.array(labels)

print("\nLoading dataset...")
X, y = load_data(CSV_PATH, IMG_BASE)

if len(X) == 0:
    print("ERROR: No images loaded.")
    exit()

# =========================
# NORMALIZE
# =========================
X = X.astype('float32') / 255.0

# =========================
# SHUFFLE
# =========================
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# =========================
# TRAIN / TEST SPLIT
# =========================
split   = int(0.8 * len(X))
X_train = X[:split];  X_test = X[split:]
y_train = y[:split];  y_test = y[split:]

print(f"\n  Training images : {len(X_train)}")
print(f"  Testing images  : {len(X_test)}")

# =========================
# BUILD CNN
# =========================
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu'),

    # Fully connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================
print("\nStarting training...")
print("This will take 10-30 minutes on CPU.\n")

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# =========================
# RESULTS
# =========================
train_acc = history.history['accuracy'][-1]     * 100
val_acc   = history.history['val_accuracy'][-1] * 100

print(f"\nFinal Training Accuracy   : {train_acc:.2f}%")
print(f"Final Validation Accuracy : {val_acc:.2f}%")

if val_acc >= 90:
    print("Status : EXCELLENT - Model is ready to use.")
elif val_acc >= 75:
    print("Status : GOOD - Acceptable for project submission.")
elif val_acc >= 60:
    print("Status : AVERAGE - Consider training more epochs.")
else:
    print("Status : LOW - Check dataset.")

# =========================
# SAVE MODEL
# =========================
save_path = os.path.join(BASE, 'models', 'sign_model.h5')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

print(f"\nModel saved to : {save_path}")
print("=" * 55)
print("Next step: python main.py")
print("=" * 55)