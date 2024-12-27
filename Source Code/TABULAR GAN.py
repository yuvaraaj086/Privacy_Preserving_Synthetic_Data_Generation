from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from tabgan.sampler import SamplerGAN

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
