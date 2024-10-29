import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb

# Assuming 'heartrate' is a feature and 'state' is the label column
X = df[['ID', 'x-axis', 'y-axis', 'z-axis', 'Heartbeat']]
y = df['state']

# Convert categorical labels to numerical labels if needed (not necessary for XGBoost)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (assuming the 'Heartbeat' column is the time series)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Create an LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
history = lstm_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predict on the test data using the LSTM model
y_pred_lstm_probs = lstm_model.predict(X_test_reshaped)
y_pred_lstm = (y_pred_lstm_probs > 0.5).astype(int).flatten()

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

# Train the XGBoost classifier on the training set
xgb_classifier.fit(X_train_scaled, y_train)

# Predict on the test data using XGBoost
y_pred_xgb = xgb_classifier.predict(X_test_scaled)

# Ensemble by averaging the predictions
ensemble_predictions = (y_pred_lstm_probs.flatten() + y_pred_xgb) / 2
ensemble_labels = (ensemble_predictions > 0.5).astype(int)

# Display classification report and accuracy for LSTM
print("\nLSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))
print("LSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))

# Display classification report and accuracy for XGBoost
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Display classification report and accuracy for ensemble
print("\nEnsemble Classification Report:\n", classification_report(y_test, ensemble_labels))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_labels))

# Plot validation loss and accuracy
plt.figure(figsize=(10, 5))

# Plot validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
