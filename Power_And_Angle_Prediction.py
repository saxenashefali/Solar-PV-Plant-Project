import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pvlib
import datetime
import warnings
from flask import Flask, jsonify, request 

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.precision',3)

# read the required data from csv
generation_data = pd.read_csv('Plant_2_Generation_Data.csv')
weather_data = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')

# adjusting date and time 
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# merging above two dataframes 
df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# adding separate time and date columns
df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
df_solar["TIME"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.time
df_solar['DAY'] = pd.to_datetime(df_solar['DATE_TIME']).dt.day
df_solar['MONTH'] = pd.to_datetime(df_solar['DATE_TIME']).dt.month

# Use isocalendar() to get ISO calendar year, week number, and weekday
df_solar['YEAR'],df_solar['WEEK'], _ = pd.to_datetime(df_solar['DATE_TIME']).dt.isocalendar().values.T

# add hours and minutes for ml models
df_solar['HOURS'] = pd.to_datetime(df_solar['TIME'], format='%H:%M:%S').dt.hour
df_solar['MINUTES'] = pd.to_datetime(df_solar['TIME'], format='%H:%M:%S').dt.minute
df_solar['TOTAL MINUTES PASS'] = df_solar['MINUTES'] + df_solar['HOURS']*60

# add date as string column
df_solar["DATE_STRING"] = df_solar["DATE"].astype(str) # add column with date as string
df_solar["HOURS"] = df_solar["HOURS"].astype(str)
df_solar["TIME"] = df_solar["TIME"].astype(str)

encoder = LabelEncoder()
df_solar['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df_solar['SOURCE_KEY'])


# Solar Power Plant Inverter Efficiency Calculation
solar_dc_power = df_solar[df_solar['DC_POWER'] > 0]['DC_POWER'].values
solar_ac_power = df_solar[df_solar['AC_POWER'] > 0]['AC_POWER'].values

solar_plant_eff = (np.max(solar_ac_power)/np.max(solar_dc_power ))*100


AC_list=[]
for i in df_solar['AC_POWER']:
    if i>0:
        AC_list.append(i)
AC_list


# Here we take all nonzero DC values and plot them on histogram
DC_list=[]
for i in df_solar['DC_POWER']:
    if i>0:
        DC_list.append(i)
DC_list
DC_list.sort()
DC_list.reverse()


AC_list.sort()
DC_list.sort()


# Machine learning model capable of forecasting solar energy production

df2 = df_solar.copy()
X = df2[['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER']]
y = df2['AC_POWER']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=21)






# def linear_regression_model():
#     try:
#         print('linear regression model')
#         lr = LinearRegression()
#         lr.fit(X_train,y_train)
#         y_pred_lr = lr.predict(X_test)
#         R2_Score_lr = round(r2_score(y_pred_lr,y_test) * 100, 2)
#         print("R2 Score : ",R2_Score_lr,"%")
#         return R2_Score_lr


#     except Exception as error:
#         print('Error : ',error)

def randomForest_regression_model():
    try:
        print('randomForest regression model')
        rfr = RandomForestRegressor()
        rfr.fit(X_train,y_train)
        y_pred_rfr = rfr.predict(X_test)
        R2_Score_rfr = round(r2_score(y_pred_rfr,y_test) * 100, 2)

        print("R2 Score : ",R2_Score_rfr,"%")
        return R2_Score_rfr

    except Exception as error:
        print('Error : ',error)

# def decisionTree_regression_model():
#     try:
#         print('decisionTree regression model')
#         dtr = DecisionTreeRegressor()
#         dtr.fit(X_train,y_train)

#         y_pred_dtr = dtr.predict(X_test)
#         R2_Score_dtr = round(r2_score(y_pred_dtr,y_test) * 100, 2)

#         print("R2 Score : ",R2_Score_dtr,"%")
#         return R2_Score_dtr

#     except Exception as error:
#         print('Error : ',error)


def hourly_power_prediction():
    try:
        print('hourly power prediction')
        X = df_solar[['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']]
        y = df_solar['AC_POWER']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)

        current_datetime = pd.to_datetime(df_solar['DATE_TIME'].iloc[-1])  # Get the most recent datetime
        next_24_hours = [current_datetime + datetime.timedelta(hours=i+1) for i in range(24)]  # Generate datetime for next 24 hours
        next_24_hours_features = df_solar.iloc[-1][['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']].values.reshape(1, -1)
        next_24_hours_predictions = []
        for i in range(24):
            prediction = rf_model.predict(next_24_hours_features)
            next_24_hours_predictions.append(prediction[0])
            next_24_hours_features[0][-1] = prediction[0]  # Update DC_POWER with the predicted value

        
        return next_24_hours_predictions

    except Exception as error:
        print('Error',error)


def weekly_power_prediction():
    try:
        print('weekly power prediction')
        X = df_solar[['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']]
        y = df_solar['AC_POWER']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        current_datetime = pd.to_datetime(df_solar['DATE_TIME'].iloc[-1])  # Get the most recent datetime

        next_7_days = [current_datetime + datetime.timedelta(days=i+1) for i in range(7)]  # Generate datetime for next 7 days
        next_7_days_features = df_solar.iloc[-1][['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']].values.reshape(1, -1)
        next_7_days_predictions = []
        for i in range(7 * 24):
            prediction = rf_model.predict(next_7_days_features)
            next_7_days_predictions.append(prediction[0])
            next_7_days_features[0][-1] = prediction[0]  # Update DC_POWER with the predicted value
            if (i + 1) % 24 == 0:
                next_7_days_features[0][0] += 1  # Increment DAILY_YIELD for the next day

        next_7_days_predictions_daily = [np.mean(next_7_days_predictions[i:i+24]) for i in range(0, len(next_7_days_predictions), 24)]

        return next_7_days_predictions_daily

    except Exception as error:
        print('Error',error)



# SOLAR ANGLE PREDICTOION 


# Function to calculate solar elevation angle
def calculate_solar_elevation(latitude, longitude, date_time):
    try:
        location = pvlib.location.Location(latitude, longitude)
        solar_position = location.get_solarposition(date_time)
        solar_elevation = solar_position.elevation[0]
        return solar_elevation
    except Exception as error:
        print('error',error)

# Function to determine optimal tilt angle based on latitude
def optimal_tilt_angle(latitude):
    return latitude  # Assuming optimal tilt angle is equal to latitude for fixed solar panels

# Function to adjust optimal tilt angle based on seasonal variation
def adjust_tilt_angle_for_season(latitude, month):
    if month in [12, 1, 2]:  # Winter months (December, January, February)
        return latitude + 10  # Increase tilt angle by 10 degrees for better winter sun capture
    elif month in [6, 7, 8]:  # Summer months (June, July, August)
        return latitude - 10  # Decrease tilt angle by 10 degrees for better summer ventilation
    else:
        return latitude  # Use the original tilt angle for other months

# Calculate daily optimal tilt angle
def daily_optimal_tilt_angle(latitude, longitude, date_time):
    try:
        solar_elevation = calculate_solar_elevation(latitude, longitude, date_time)
        month = date_time.month
        tilt_angle = optimal_tilt_angle(latitude)
        tilt_angle = adjust_tilt_angle_for_season(tilt_angle, month)
        return [tilt_angle , solar_elevation]
    except Exception as error:
        print('error', error)




def optimal_solar_panel_angles_seasonal(latitude,longitude,start_date,end_date):
    try:
        print('optimal_solar_panel_angles_seasonal')
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        solar_panel_angles = []
        dates = pd.date_range(start_date, end_date)
        for date in dates:
            angle = daily_optimal_tilt_angle(latitude, longitude, date)
            obj ={}
            obj['Date'] = date
            obj['Tilt_Angle'] = angle[0]
            obj['Elevation'] = angle[1]
            solar_panel_angles.append(obj)

        return solar_panel_angles
    
    except Exception as error:
        print('Error : ',error)



def optimal_solar_panel_angles_daily(latitude,longitude,date):
    try:
        print('optimal_solar_panel_angles_seasonal')
        start_date = pd.Timestamp(date)
        end_date = start_date + pd.Timedelta(days=1)
        solar_panel_angles = []
        dates = pd.date_range(start_date, end_date, freq='H')
        for date_time in dates:
            angle = daily_optimal_tilt_angle(latitude, longitude, date_time)
            obj ={}
            obj['Date'] = date_time
            obj['Tilt_Angle'] = angle[0]
            obj['Elevation'] = angle[1]
            solar_panel_angles.append(obj)

        return solar_panel_angles

    except Exception as error:
        print('Error : ',error)




# creating a Flask app 
app = Flask(__name__) 

@app.route('/get_hourly_solar_power_prediction', methods = ['GET', 'POST']) 
def daily_solar_power_prediction(): 
    hourly_power_data = hourly_power_prediction()
    return jsonify({'data': hourly_power_data}) 

@app.route('/get_weekly_solar_power_prediction', methods = ['GET', 'POST']) 
def weekly_solar_power_prediction(): 
    weekly_power_data = weekly_power_prediction()
    return jsonify({'data': weekly_power_data}) 


@app.route('/get_seasonal_optimal_solar_angle', methods = ['GET', 'POST']) 
def seasonal_optimal_angle(): 
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    seasonal_optimal_angles = optimal_solar_panel_angles_seasonal(latitude,longitude,start_date,end_date)
    return jsonify({'data': seasonal_optimal_angles}) 


@app.route('/get_daily_optimal_solar_angle', methods = ['GET', 'POST']) 
def daily_optimal_angle(): 
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    date = request.json.get('date')
    daily_optimal_angles = optimal_solar_panel_angles_daily(latitude,longitude,date)
    return jsonify({'data': daily_optimal_angles}) 


# driver function 
if __name__ == '__main__': 

    app.run(debug = True) 
