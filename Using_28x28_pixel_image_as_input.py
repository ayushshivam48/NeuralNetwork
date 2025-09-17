import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

# ============================================================
# 1️⃣ Data Loading & Preprocessing
# ============================================================
print("\n=== Step 1: Data Loading & Preprocessing ===")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)
print("Labels range:", y_train.min(), "to", y_train.max())

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("Sample pixel value after normalization:", x_train[0, 0, 0])

# ============================================================
# 2️⃣ Model Building
# ============================================================
print("\n=== Step 2: Model Building ===")
model = Sequential([
    Input(shape=(28, 28)),                    # Input: 28x28 image
    Flatten(),                                # Flatten -> 784
    Dense(1024, activation='relu'),           # Hidden Layer 1
    Dense(1024, activation='relu'),           # Hidden Layer 2
    Dense(512, activation='relu'),            # Hidden Layer 3
    Dense(10, activation='softmax')           # Output: 10 classes
])

model.summary()

# ============================================================
# 3️⃣ Model Compilation & Training
# ============================================================
print("\n=== Step 3: Compilation & Training ===")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,              # you can increase epochs for better accuracy
    batch_size=128,
    verbose=2
)

print("Training history keys:", history.history.keys())

# ============================================================
# 4️⃣ Evaluation & Prediction
# ============================================================
print("\n=== Step 4: Evaluation & Prediction ===")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Predict first 5 test images
predictions = model.predict(x_test[:5])
print("Predicted labels:", np.argmax(predictions, axis=1))
print("True labels     :", y_test[:5])

