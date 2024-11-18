import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('https://drive.google.com/uc?id=19903lXYiKFUwB6oVn8tWFL4KsraZ0y-p')

# Features and target
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
target = "Spending Score (1-100)"

# One hot encoding categorical features
df_with_dummies = pd.get_dummies(df[features])

# Splitting the data
X = df_with_dummies.drop(columns=[target])
y = df_with_dummies[target]

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Building the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluating the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on test set: {mae}")

# Making predictions
y_pred = model.predict(X_test)

# Saving the model
model.save('ann_model.h5')

# Plotting the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
