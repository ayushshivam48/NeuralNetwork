# ==========================================
# 📌 Spam Email Detection using ANN
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 1. Load Dataset
# ==========================================
# Download dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Try an alternative source for the dataset by reading directly from the URL
try:
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']

    if df.empty:
        print("Error: The dataset loaded is empty.")
        df = None # Ensure df is None if the loaded data is empty

except Exception as e:
    print(f"Error loading data directly from URL: {e}")
    df = None # Ensure df is None if there's an exception during loading


if df is None:
    print("Failed to load the dataset.")
else:
    print("Sample Data:\n", df.head())

    # Encode labels: ham=0, spam=1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # ==========================================
    # 2. Preprocessing - TF-IDF
    # ==========================================
    X = df['message']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocab size
    X = vectorizer.fit_transform(X).toarray()

    print("Shape of feature matrix:", X.shape)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. ANN Model
    # ==========================================
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # ==========================================
    # 4. Training
    # ==========================================
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ==========================================
    # 5. Evaluation
    # ==========================================
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy/Loss Graphs
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    # ==========================================
    # 6. Test with Custom Input
    # ==========================================
    def predict_message(msg):
        msg_vec = vectorizer.transform([msg]).toarray()
        pred = model.predict(msg_vec)[0][0]
        return "Spam 🚨" if pred > 0.5 else "Ham ✅"

    print("\n🔮 Prediction Demo:")
    print("Message: 'Congratulations! You have won a free lottery ticket'")
    print("Result:", predict_message("Congratulations! You have won a free lottery ticket"))

    print("Message: 'Hey, are we still on for dinner tonight?'")
    print("Result:", predict_message("Hey, are we still on for dinner tonight?"))
