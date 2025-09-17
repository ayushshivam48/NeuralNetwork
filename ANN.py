import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Example dummy dataset (replace with your dataset)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.rand(1000, 1)   # target values

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential([
    Input(shape=(X.shape[1],)),       # Input layer (10 features)
    Dense(64, activation='relu'),     # 1st hidden layer
    Dense(32, activation='relu'),     # 2nd hidden layer
    Dense(1, activation='linear')     # Output layer (for regression)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mse']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model on test data
test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_mse:.4f}")
