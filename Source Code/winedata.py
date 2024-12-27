import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "C:\\Users\\ACER\\Downloads\\wine.csv"
wine_data = pd.read_csv(file_path)

# Set the target column
target_column = "Proline"

# Split features and target
X = wine_data.drop(target_column, axis=1)
y = wine_data[target_column]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate synthetic dataset (for demonstration purposes)
# You can replace this with your actual synthetic data generation method
n_samples_synthetic = len(X_train)
n_features_synthetic = X_train.shape[1]
X_synthetic = pd.DataFrame({'feature_1': [0]*n_samples_synthetic, 'feature_2': [1]*n_samples_synthetic})  # Example synthetic features
y_synthetic = pd.Series([0]*n_samples_synthetic)  # Example synthetic target variable

# Splitting synthetic data into train and test sets
X_synthetic_train, X_synthetic_test, y_synthetic_train, y_synthetic_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Train Random Forest classifier on real data
rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
rf_real.fit(X_train, y_train)

# Train Random Forest classifier on synthetic data
rf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
rf_synthetic.fit(X_synthetic_train, y_synthetic_train)

# Predictions on real test data
y_pred_real = rf_real.predict(X_test)

# Predictions on synthetic test data
y_pred_synthetic = rf_synthetic.predict(X_synthetic_test)

# Calculate accuracy scores
accuracy_real = accuracy_score(y_test, y_pred_real)
accuracy_synthetic = accuracy_score(y_synthetic_test, y_pred_synthetic)

# Calculate RMSE for real data
rmse_real = np.sqrt(mean_squared_error(y_test, y_pred_real))

# Calculate RMSE for synthetic data
rmse_synthetic = np.sqrt(mean_squared_error(y_synthetic_test, y_pred_synthetic))

# Print accuracy and RMSE scores
print("Accuracy for Real Data:", accuracy_real)
print("Accuracy for Synthetic Data:", accuracy_synthetic)
print("RMSE for Real Data:", rmse_real)
print("RMSE for Synthetic Data:", rmse_synthetic)

# Plotting the results
plt.bar(['Real Data', 'Synthetic Data'], [accuracy_real, accuracy_synthetic], color=['blue', 'orange'])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between Real and Synthetic Data')
plt.ylim(0, 1)  # Set y-axis limits
plt.show()
