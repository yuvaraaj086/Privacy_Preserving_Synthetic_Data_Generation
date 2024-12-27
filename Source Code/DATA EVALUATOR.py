import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

def load_data(real_file_path, synthetic_file_path):
    real_data = pd.read_csv(real_file_path)
    synthetic_data = pd.read_excel(synthetic_file_path)
    return real_data, synthetic_data

def preprocess_data(data):
    # Replace '?' with NaN
    data.replace('?', pd.NA, inplace=True)
    # Convert columns to numeric
    data = data.apply(pd.to_numeric, errors='ignore')
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

def evaluate(real_data, synthetic_data, target_columns):
    evaluation_results = {}
    for target_column in target_columns:
        # Extract target columns
        real_target = real_data[target_column]
        synthetic_target = synthetic_data[target_column]

        # Discretize continuous values into bins
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        real_target_bins = discretizer.fit_transform(real_target.values.reshape(-1, 1))
        synthetic_target_bins = discretizer.transform(synthetic_target.values.reshape(-1, 1))

        # Encode discretized target column
        label_encoder = LabelEncoder()
        real_target_encoded = label_encoder.fit_transform(real_target_bins)
        synthetic_target_encoded = label_encoder.transform(synthetic_target_bins)

        # Generate classification report and confusion matrix
        class_report = classification_report(real_target_encoded, synthetic_target_encoded, output_dict=True)
        confusion_mat = confusion_matrix(real_target_encoded, synthetic_target_encoded)

        # Store evaluation results
        evaluation_results[target_column] = {
            'classification_report': class_report,
            'confusion_matrix': confusion_mat
        }

    return evaluation_results

# Specify file paths for real and synthetic datasets
real_file_path = r"C:\Users\91934\Downloads\auto-mpg.csv"
synthetic_file_path = r"C:\Users\91934\OneDrive\Desktop\synthetic_data.xlsx"

# Specify the target columns for evaluation
target_columns = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin"]

# Load real and synthetic datasets
real_data, synthetic_data = load_data(real_file_path, synthetic_file_path)

# Preprocess data
real_data = preprocess_data(real_data)
synthetic_data = preprocess_data(synthetic_data)

# Perform evaluation
evaluation_results = evaluate(real_data, synthetic_data, target_columns)

# Display evaluation results
for target_column, results in evaluation_results.items():
    print(f"Evaluation results for {target_column}:")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("\n")
