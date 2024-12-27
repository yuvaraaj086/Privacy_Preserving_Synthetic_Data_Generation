from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
from tabgan.sampler import SamplerGAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", na_values=['NA', '?'])

# Select columns for training
COLS_USED = ['cylinders', 'displacement', 'horsepower', 'weight', 
          'acceleration', 'year', 'origin','mpg']
COLS_TRAIN = ['cylinders', 'displacement', 'horsepower', 'weight', 
          'acceleration', 'year', 'origin']

df = df[COLS_USED]

# Handle missing values
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    df.drop("mpg", axis=1),
    df["mpg"],
    test_size=0.20,
    random_state=42,
)

# Build the neural network
model = Sequential()
model.add(Dense(50, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Set up early stopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto',
                        restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          callbacks=[monitor], verbose=2, epochs=1000)

# Evaluate the model on the test set
pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred, y_test))
print("Final score (RMSE) on test set: {}".format(score))

# Generate Synthetic Data
sampler = SamplerGAN()
synthetic_data = sampler.generate_data(x_train, y_train, None, only_generated_data=True)  # Pass x_train and y_train instead of df and "mpg"
xgan, ygan = synthetic_data  # Tuple unpacking

# Ensure xgan is preprocessed in the same way as x_train
xgan = xgan.values  # Convert DataFrame to numpy array
# If required, perform additional preprocessing on xgan (e.g., scaling)

# Evaluate the model on the synthetic data
pred_synthetic = model.predict(xgan)
score_synthetic = np.sqrt(metrics.mean_squared_error(pred_synthetic, ygan))
print("Final score (RMSE) on synthetic data: {}".format(score_synthetic))

# Save the generated synthetic data to an Excel file
synthetic_df = pd.DataFrame(xgan, columns=COLS_TRAIN)  # Create DataFrame from the synthetic data
synthetic_df.to_excel("synthetic_data.xlsx", index=False)  # Save DataFrame to Excel file
print("\nGenerated Synthetic Data saved to 'synthetic_data.xlsx'")

# Plotting the RMSE scores
plt.bar(['Test Set', 'Synthetic Data'], [score, score_synthetic], color=['blue', 'green'])
plt.title('RMSE Scores Comparison')
plt.xlabel('Dataset')
plt.ylabel('RMSE Score')
plt.show()

# Binning the regression targets into classes
num_bins = 5  # Choose the number of bins
bin_edges = np.histogram_bin_edges(y_train, bins=num_bins)

# Convert the regression targets into classes
y_train_classes = np.digitize(y_train, bin_edges)
y_test_classes = np.digitize(y_test, bin_edges)
y_synthetic_classes = np.digitize(ygan, bin_edges)  # Assuming ygan contains the synthetic labels

# Train a classification model on the real dataset
clf_real = RandomForestClassifier()  # Or any classifier of your choice
clf_real.fit(x_train, y_train_classes)

# Evaluate the classification model on the test set
y_pred_real_classes = clf_real.predict(x_test)

# Calculate precision, recall, and accuracy for the real dataset
precision_real = precision_score(y_test_classes, y_pred_real_classes, average='weighted')
recall_real = recall_score(y_test_classes, y_pred_real_classes, average='weighted')
accuracy_real = accuracy_score(y_test_classes, y_pred_real_classes)

print("Precision (Real Dataset):", precision_real)
print("Recall (Real Dataset):", recall_real)
print("Accuracy (Real Dataset):", accuracy_real)

# Train a classification model on the synthetic dataset
clf_synthetic = RandomForestClassifier()  # Or any classifier of your choice
clf_synthetic.fit(xgan, y_synthetic_classes)

# Evaluate the classification model on the test set
y_pred_synthetic_classes = clf_synthetic.predict(x_test)

# Calculate precision, recall, and accuracy for the synthetic dataset
precision_synthetic = precision_score(y_test_classes, y_pred_synthetic_classes, average='weighted')
recall_synthetic = recall_score(y_test_classes, y_pred_synthetic_classes, average='weighted')
accuracy_synthetic = accuracy_score(y_test_classes, y_pred_synthetic_classes)

print("Precision (Synthetic Dataset):", precision_synthetic)
print("Recall (Synthetic Dataset):", recall_synthetic)
print("Accuracy (Synthetic Dataset):", accuracy_synthetic)
