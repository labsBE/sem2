# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


# ==============================
# 2. LOAD DATASET
# ==============================
data = pd.read_csv('housing_data.csv')

print("First 5 rows:")
print(data.head())


# ==============================
# 3. DATA PREPROCESSING
# ==============================
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

print("\nAfter Handling Missing Values:")
print(data.isnull().sum())


# ==============================
# 4. DATA ANALYSIS
# ==============================
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())


# ==============================
# 5. VISUALIZATION
# ==============================

# Distribution plot
sns.histplot(data['MEDV'], kde=True)
plt.title("Distribution of House Prices")
plt.show()

# Boxplot
sns.boxplot(x=data['MEDV'])
plt.title("Boxplot of House Prices")
plt.show()

# Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# 6. SCATTER PLOTS
# ==============================
features = ['LSTAT', 'RM', 'PTRATIO']

plt.figure(figsize=(18,5))
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    plt.scatter(data[col], data['MEDV'])
    plt.xlabel(col)
    plt.ylabel("House Price")
    plt.title(f"{col} vs MEDV")

plt.show()


# ==============================
# 7. SPLIT DATA
# ==============================
X = data.drop('MEDV', axis=1)
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 8. FEATURE SCALING
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 9. LINEAR REGRESSION MODEL
# ==============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n===== Linear Regression Results =====")
print("RMSE:", rmse_lr)
print("MAE:", mae_lr)
print("R2 Score:", r2_lr)


# ==============================
# 10. NEURAL NETWORK MODEL
# ==============================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.1,
    verbose=1
)


# ==============================
# 11. PLOT TRAINING RESULTS
# ==============================

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Curve")
plt.show()

# MAE plot
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title("MAE Curve")
plt.show()


# ==============================
# 12. EVALUATE NEURAL NETWORK
# ==============================
loss, mae = model.evaluate(X_test, y_test)

y_pred_nn = model.predict(X_test)

rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

print("\n===== Neural Network Results =====")
print("RMSE:", rmse_nn)
print("MAE:", mae)
print("R2 Score:", r2_nn)


# ==============================
# 13. MODEL COMPARISON
# ==============================
print("\n===== Model Comparison =====")
print("Linear Regression RMSE:", rmse_lr)
print("Neural Network RMSE:", rmse_nn)


# ==============================
# 14. PREDICTION ON NEW DATA
# ==============================
new_data = np.array([[0.1, 10.0, 5.0, 0, 0.4, 6.0, 50, 6.0, 1, 400, 20, 300, 10]])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("\nPredicted House Price:", prediction[0][0])
