# ============================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')


# ============================================
# 2. LOAD MNIST DATASET
# ============================================
# MNIST contains handwritten digits (0–9)
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# ============================================
# 3. VISUALIZE SAMPLE IMAGE
# ============================================
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()


# ============================================
# 4. CHECK DATA SHAPE
# ============================================
print("X_train shape:", X_train.shape)  # (60000, 28, 28)
print("y_train shape:", y_train.shape)  # (60000,)
print("X_test shape:", X_test.shape)    # (10000, 28, 28)
print("y_test shape:", y_test.shape)    # (10000,)


# ============================================
# 5. RESHAPE DATA (FLATTEN IMAGES)
# ============================================
# Convert 28x28 images into 1D vector of size 784
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


# ============================================
# 6. NORMALIZE DATA
# ============================================
# Scale pixel values from (0–255) → (0–1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# ============================================
# 7. ONE-HOT ENCODING FOR LABELS
# ============================================
# Convert labels into binary class matrix
num_classes = 10

y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]


# ============================================
# 8. BUILD NEURAL NETWORK MODEL
# ============================================
model = Sequential()

# First hidden layer
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))  # Prevent overfitting

# Second hidden layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Output layer (10 classes)
model.add(Dense(num_classes, activation='softmax'))


# ============================================
# 9. COMPILE MODEL
# ============================================
model.compile(
    loss='categorical_crossentropy',   # For multi-class classification
    optimizer=RMSprop(),
    metrics=['accuracy']
)


# ============================================
# 10. TRAIN MODEL
# ============================================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)


# ============================================
# 11. EVALUATE MODEL
# ============================================
score = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss:", score[0])
print("Test Accuracy:", score[1])


# ============================================
# 12. PLOT TRAINING PERFORMANCE
# ============================================

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Model Loss")
plt.show()


# ============================================
# 13. PREDICT SAMPLE OUTPUT
# ============================================
predictions = model.predict(X_test)

# Convert probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Display first prediction
print("\nPredicted Label:", predicted_labels[0])
print("Actual Label:", true_labels[0])
