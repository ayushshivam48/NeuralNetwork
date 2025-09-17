# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 2. Load Dataset (Sample UCI Credit Dataset)
# ==========================================
# You can replace this link with your dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" # Using a different valid URL
df = pd.read_excel(url, header=1) # The dataset is in excel format and has a header row

print("✅ Dataset Loaded Successfully")
print(df.head())

# ==========================================
# 3. Data Preprocessing
# ==========================================
# Handle categorical variables
le = LabelEncoder()
# The new dataset has different column names and types.
# Need to identify and handle categorical columns in the new dataset.
# Assuming the new dataset has columns like 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0' to 'PAY_6' as categorical.
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])


# The new dataset might not have missing values in the same way as the previous one.
# Let's check for missing values and handle them appropriately for the new dataset.
print("\nMissing values before handling:")
print(df.isnull().sum())

# For simplicity, we will drop rows with any missing values.
df.dropna(inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Features (X) and Target (y)
# The target variable in this dataset is 'default.payment.next.month'
X = df.drop(columns=['default payment next month', 'ID'])   # Remove ID column as well
y = df['default payment next month']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("🔹 Data Preprocessing Done")

# ==========================================
# 4. Build ANN Model
# ==========================================
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🔹 Model Summary")
model.summary()

# ==========================================
# 5. Train Model
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# ==========================================
# 6. Evaluate Model
# ==========================================
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# 7. Training Visualization
# ==========================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()
