# from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge
# Assume df is your cleaned and preprocessed weather data stored in a pandas DataFrame
weather_data = pd.read_csv("dataset.csv")
print(weather_data)
# weather_data['DATE'] = weather_data['YEAR'].astype(str) + "-" + weather_data['MO'].astype(str) + "-" + weather_data[
#     "DY"].astype(str)
# weather_data = weather_data.set_index('DATE')
# # print(dataset.index)
# weather_data.index = pd.to_datetime(weather_data.index)
# # print(dataset.index)
# dataset = weather_data.loc[:, ['T2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'PS']]
# # dataset["PRECTOTCORR"].plot()
# weather_data["Temperature"] = weather_data.shift(0)["T2M"]
# weather_data['Humidity'] = weather_data.shift(0)["RH2M"]
# weather_data['Prediction'] = weather_data.shift(0)["PRECTOTCORR"]
# weather_data['Wind_Speed'] = weather_data.shift(0)["WS10M"]
# weather_data['Atmospheric_pressure'] = weather_data.shift(0)["PS"]
# # dataset['target']=dataset.shift(-1)["PRECTOTCORR"]
# weather_data = weather_data.loc[:, ['Temperature', 'Humidity', 'Prediction', 'Wind_Speed', 'Atmospheric_pressure']]
# print(weather_data)
