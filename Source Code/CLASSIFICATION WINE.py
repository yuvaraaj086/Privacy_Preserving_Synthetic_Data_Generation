import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load synthetic dataset from CSV
synthetic_data = pd.read_csv(r"C:\Users\91934\OneDrive\Desktop\ALCOHOL.csv")

# Drop rows with NaN values
synthetic_data.dropna(inplace=True)

# Define initial threshold for wine quality
initial_threshold = 6  # Adjust the initial threshold as needed

# Convert continuous target variable to binary classes using the initial threshold
synthetic_data["Wine_quality"] = (synthetic_data["Alcohol"] >= initial_threshold).astype(int)

# Check if classes are imbalanced after applying the initial threshold
class_distribution = synthetic_data["Wine_quality"].value_counts()

# If the distribution is imbalanced, adjust the threshold
if len(class_distribution) < 2:
    print("Warning: Imbalanced classes. Adjusting threshold...")
    # Adjust the threshold based on the median value of the target variable
    new_threshold = synthetic_data["Alcohol"].median()
    synthetic_data["Wine_quality"] = (synthetic_data["Alcohol"] >= new_threshold).astype(int)
    print("New threshold:", new_threshold)
else:
    print("Classes are balanced.")

# Update X and y with the new threshold
X_synthetic = synthetic_data.drop(columns=["Alcohol", "Wine_quality"])  # Adjust column names accordingly
y_synthetic = synthetic_data["Wine_quality"]  # Adjust column name accordingly

# Load real dataset from CSV
real_data = pd.read_csv(r"C:\Users\91934\Downloads\SYNTHETIC DATA WINE.csv")

# Drop rows with NaN values
real_data.dropna(inplace=True)

# Convert continuous target variable to binary classes using the new threshold
real_data["Wine_quality"] = (real_data["Alcohol"] >= new_threshold).astype(int)

X_real = real_data.drop(columns=["Alcohol", "Wine_quality"])  # Adjust column names accordingly
y_real = real_data["Wine_quality"]  # Adjust column name accordingly

# Split datasets into training and testing sets
X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# Train logistic regression model on synthetic dataset
model_synthetic = LogisticRegression()
model_synthetic.fit(X_train_synthetic, y_train_synthetic)

# Train logistic regression model on real dataset
model_real = LogisticRegression()
model_real.fit(X_train_real, y_train_real)

# Predictions on synthetic dataset
y_pred_synthetic = model_synthetic.predict(X_test_synthetic)
# Predictions on real dataset
y_pred_real = model_real.predict(X_test_real)

# Calculate metrics for synthetic dataset
accuracy_synthetic = accuracy_score(y_test_synthetic, y_pred_synthetic)
precision_synthetic = precision_score(y_test_synthetic, y_pred_synthetic)
recall_synthetic = recall_score(y_test_synthetic, y_pred_synthetic)
print("Metrics for Synthetic Dataset:")
print("Accuracy:", accuracy_synthetic)
print("Precision:", precision_synthetic)
print("Recall:", recall_synthetic)

# Calculate metrics for real dataset
accuracy_real = accuracy_score(y_test_real, y_pred_real)
precision_real = precision_score(y_test_real, y_pred_real)
recall_real = recall_score(y_test_real, y_pred_real)
print("\nMetrics for Real Dataset:")
print("Accuracy:", accuracy_real)
print("Precision:", precision_real)
print("Recall:", recall_real)
