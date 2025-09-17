# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 2. Load Dataset
# ==========================================
# Example dataset (Daily Delhi Climate dataset)
# Using an alternative source for the dataset
url = "Weather Data.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded")
print(df.head())

# For simplicity, let's forecast temperature
df = df.dropna()
df = df[['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']] # Updated column names to match the new dataset

# Features (X) and Target (y = temperature)
X = df[['Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']] # Updated feature columns
y = df['Temp_C'] # Updated target column

# ==========================================
# 3. Preprocessing
# ==========================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("🔹 Data Preprocessing Done")

# ==========================================
# 4. Build ANN Model
# ==========================================
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')  # Regression output
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

print("🔹 Model Summary")
model.summary()

# ==========================================
# 5. Train Model
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=1
)

# ==========================================
# 6. Evaluate Model
# ==========================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ==========================================
# 7. Visualization
# ==========================================
# Training Accuracy/Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Predictions vs Actual
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Actual Temperature', marker='o')
plt.plot(y_pred[:100], label='Predicted Temperature', marker='x')
plt.legend()
plt.title("Actual vs Predicted Temperature (Sample 100)")
plt.show()
