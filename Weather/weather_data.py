
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv("weather_data.csv")

# Create a new date column and set it as the index
dataset['DATE'] = dataset['YEAR'].astype(str) + "-" + dataset['MO'].astype(str) + "-" + dataset["DY"].astype(str)
dataset = dataset.set_index('DATE')
dataset.index = pd.to_datetime(dataset.index)

# Select relevant columns
dataset = dataset.loc[:, ['T2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'PS']]

# Rename columns for clarity
dataset.rename(columns={'T2M': 'Temperature', 'RH2M': 'Humidity', 'PRECTOTCORR': 'Prediction',
                        'WS10M': 'Wind_Speed', 'PS': 'Atmospheric_pressure'}, inplace=True)

# Fill missing values in 'Prediction' with mode
# precip_type_mode = dataset['Prediction'].mode()[0]
# dataset['Prediction'].fillna(precip_type_mode, inplace=True)

# Encode 'Prediction' column
label_encoder = LabelEncoder()
dataset['Prediction'] = label_encoder.fit_transform(dataset['Prediction'])

# Define features and target variable
features = dataset[['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']]
target = dataset['Prediction']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred = decision_tree.predict(X_test)

# Generate classification report and confusion matrix
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report_result)
print("Confusion Matrix:")
print(confusion_matrix_result)


# Function to predict new data
def predict_new_data(new_data):
    """
    Predict the precipitation type for new data using the trained Decision Tree model.

    Parameters:
    new_data (pd.DataFrame): DataFrame containing new data with columns
                             ['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']

    Returns:
    np.array: Array containing the predicted precipitation types.
    """
    if not all(
            column in new_data.columns for column in ['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']):
        raise ValueError(
            "New data must contain the columns: 'Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure'")

    # Make predictions on new data
    predictions = decision_tree.predict(new_data[['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']])

    # Return predictions in original labels
    return label_encoder.inverse_transform(predictions)


# Example usage of the predict_new_data function
current_weather_data = {
    'Temperature': [26],
    'Humidity': [65],
    'Wind_Speed': [6],
    'Atmospheric_pressure': [1012]
}

current_weather_df = pd.DataFrame(current_weather_data)

current_predictions = predict_new_data(current_weather_df)

rain_status = [
    'No Rain'
    if pred == label_encoder.transform([current_predictions])[0]
    else 'Rain'
    for pred in current_predictions
]

print("Prediction for current weather (Rain or No Rain):")
print(rain_status)

