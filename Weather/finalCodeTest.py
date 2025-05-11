import requests
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
from pprint import pprint
import pytz
import iot_api_client as iot
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# API and configuration settings
openai_api_key = 'sk-proj-wRnW5mN5sZUAnyvue6lqT3BlbkFJ6XrCGlgmydtNnaKmhXOS'
CLIENT_ID = "VJoXtqaxLeHuxKPEX7hvT1zONijN3t9K"
CLIENT_SECRET = "O2Hy2iBsM3lTQm0usceVl9u73dim7DuoZEF9SELplkqbhDRktyIb7qRIvNRnhVgd"
THING_ID = "2764842a-7615-4a95-91e2-e08ab7bc818b"
DEVICE_ID = "230f138a-2181-4937-9e6b-09261075779c"
PROPERTY_ID_CONTROL = "2b748e9f-350f-40f5-8418-157ae0ab2e3b"
PROPERTY_ID_MOISTURE = "4a3fa865-2b96-46c3-81df-4e896ddde5ad"
PROPERTY_ID_RELAY = "68855c55-06dd-4272-ac6d-ed53bfec4d90"
PROPERTY_ID_SCHEDULE1 = "abc918f0-65c9-4426-a538-6934a94e2a55"
PROPERTY_ID_SCHEDULE2 = "f964f4e1-823c-479f-b97d-bc11f49dc4da"

telegram_token = "7400352290:AAHGPHmt9_ajLGMgPqT-Z9XNQNk4KAAXSVI"
chat_id = "IoT_Team"
api_key = "4ac53b87c2233ee8de919d51d83a4347"
CITY_NAME = 'Phnom Penh'


# Function to send a message via Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage?chat_id=@{chat_id}&text={message}"
    response = requests.post(url, data={'chat_id': chat_id, 'text': message})
    return response


# Function to fetch weather data from OpenWeatherMap API
def fetch_weather_data(api_key, city_name):
    try:
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        lon, lat = data['coord']['lon'], data['coord']['lat']
        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely&appid={api_key}'
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


# Function to train the weather prediction model
def train_weather_model(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset['DATE'] = dataset['YEAR'].astype(str) + "-" + dataset['MO'].astype(str) + "-" + dataset["DY"].astype(str)
    dataset = dataset.set_index('DATE')
    dataset.index = pd.to_datetime(dataset.index)
    dataset = dataset.loc[:, ['T2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'PS']]
    dataset.rename(columns={'T2M': 'Temperature', 'RH2M': 'Humidity', 'PRECTOTCORR': 'Prediction',
                            'WS10M': 'Wind_Speed', 'PS': 'Atmospheric_pressure'}, inplace=True)
    precip_type_mode = dataset['Prediction'].mode()[0]
    dataset['Prediction'].fillna(precip_type_mode, inplace=True)
    label_encoder = LabelEncoder()
    dataset['Prediction'] = label_encoder.fit_transform(dataset['Prediction'])
    features = dataset[['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']]
    target = dataset['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return decision_tree, label_encoder, precip_type_mode


# Function to predict rain status using the trained model
def predict_rain_status(model, label_encoder, precip_type_mode, current_weather_df):
    def predict_new_data(new_data):
        if not all(column in new_data.columns for column in
                   ['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']):
            raise ValueError(
                "New data must contain the columns: 'Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure'")
        predictions = model.predict(new_data[['Temperature', 'Humidity', 'Wind_Speed', 'Atmospheric_pressure']])
        return label_encoder.inverse_transform(predictions)

    current_predictions = predict_new_data(current_weather_df)
    rain_status = ['No Rain' if pred == label_encoder.transform([precip_type_mode])[0] else 'Rain' for pred in
                   current_predictions]
    return current_weather_df['Date'], rain_status


# Function to control Arduino relay
def control_arduino_relay(properties, thing_id, property_id_relay, action):
    property_value = {"value": action}
    properties.properties_v2_publish(thing_id, property_id_relay, property_value)


def control_arduino_motor(properties, thing_id, property_id_control, action):
    property_value = {"value": action}
    properties.properties_v2_publish(thing_id, property_id_control, property_value)


# Main function to integrate everything
def main(dataset_path, api_key, city_name):
    # Train the model
    model, label_encoder, precip_type_mode = train_weather_model(dataset_path)

    # Fetch weather data
    weather_data = fetch_weather_data(api_key, city_name)
    if not weather_data:
        return
    current_weather_df = pd.DataFrame(weather_data['hourly'])
    current_weather_df['Date'] = current_weather_df['dt']
    current_weather_df['Temperature'] = current_weather_df['temp']
    current_weather_df['Humidity'] = current_weather_df['humidity']
    current_weather_df['Wind_Speed'] = current_weather_df['wind_speed']
    current_weather_df['Atmospheric_pressure'] = current_weather_df['pressure']

    # Predict rain status
    day = []
    hour = []
    timestamps, rain_status = predict_rain_status(model, label_encoder, precip_type_mode, current_weather_df)
    # print(timestamps)
    for ts, status in zip(timestamps, rain_status):
        # for timestamp, status in zip(readable_timestamps, rain_status):
        # Extract the timestamp by splitting the input
        ts = str(ts)
        ts = int(ts)
        datetime_obj = datetime.fromtimestamp(ts)
        # Get the day and hour
        if status == 'Rain':
            day.append(datetime_obj.strftime("%A"))
            hour.append(datetime_obj.hour)

    # Convert timestamps to readable format
    readable_timestamps = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    # print(time.strftime("%c", timestamps))

    # Print predictions with timestamps
    print("Prediction for current weather (Rain or No Rain) by time:")
    for timestamp, status in zip(readable_timestamps, rain_status):
        print(f"{timestamp}: {status}")

    # Create a new API session
    oauth_client = BackendApplicationClient(client_id=CLIENT_ID)
    token_url = "https://api2.arduino.cc/iot/v1/clients/token"
    oauth = OAuth2Session(client=oauth_client)
    token = oauth.fetch_token(
        token_url=token_url,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        include_client_id=True,
        audience="https://api2.arduino.cc/iot",
    )
    print("Got a token, expires in {} seconds\n".format(token.get("expires_in")))
    client_config = iot.Configuration(host="https://api2.arduino.cc/iot")
    client_config.access_token = token.get("access_token")
    client = iot.ApiClient(client_config)
    properties = iot.PropertiesV2Api(client)

    # Read schedule and moisture data
    api_response_schedule1 = properties.properties_v2_show(THING_ID, PROPERTY_ID_SCHEDULE1)
    frm_timestamp1 = api_response_schedule1.last_value['frm']
    frm_datetime1 = datetime.fromtimestamp(frm_timestamp1)
    frm_datetime1_hour = frm_datetime1.hour

    api_response_schedule2 = properties.properties_v2_show(THING_ID, PROPERTY_ID_SCHEDULE2)
    frm_timestamp2 = api_response_schedule2.last_value['frm']
    frm_datetime2 = datetime.fromtimestamp(frm_timestamp2)
    frm_datetime2_hour = frm_datetime2.hour

    api_response_moisture = properties.properties_v2_show(THING_ID, PROPERTY_ID_MOISTURE)
    api_response_rain = properties.properties_v2_show(THING_ID, PROPERTY_ID_MOISTURE)
    # Execute scheduling logic
    while True:
        current_hour = datetime.now().hour
        current_day = datetime.now().day
        moisture_level = api_response_moisture.last_value
        rain_level = api_response_rain.last_value

        if current_hour == 24 and datetime.now().second == 1:
            for timestamp, status in zip(readable_timestamps, rain_status):
                weather_update = f"Weather Update: {timestamp:status}"
                send_telegram_message(weather_update)
        for h, d in zip(hour, day):
            if (h in range(12) and d == current_day) or (h in range(12, 24) and d == current_day):
                control_arduino_relay(properties, THING_ID, PROPERTY_ID_RELAY, False)
                control_arduino_motor(properties, THING_ID, PROPERTY_ID_CONTROL, True)
            elif (h not in range(12) and d == current_day) or (h not in range(12, 24) and d == current_day):
                control_arduino_relay(properties, THING_ID, PROPERTY_ID_RELAY, True)
                control_arduino_motor(properties, THING_ID, PROPERTY_ID_CONTROL, False)
            elif (h in range(
                    12) and current_hour > 3 + h and d == current_day and moisture_level < 100 and rain_level < 10) or (
                    h in range(12,
                               24) and current_hour == 3 + h and d == current_day and moisture_level < 100 and rain_level < 10):
                control_arduino_relay(properties, THING_ID, PROPERTY_ID_RELAY, True)
                control_arduino_motor(properties, THING_ID, PROPERTY_ID_CONTROL, False)
            else:
                control_arduino_relay(properties, THING_ID, PROPERTY_ID_RELAY, False)
                control_arduino_motor(properties, THING_ID, PROPERTY_ID_CONTROL, True)
        time.sleep(3600)  # Check every second


# Define dataset path and API details
dataset_path = "a.csv"
api_key = "4ac53b87c2233ee8de919d51d83a4347"
city_name = "Phnom Penh"

# Call the main function
main(dataset_path, api_key, city_name)
