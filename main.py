import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

data = pd.read_csv("./data/deliverytime.txt")

# basic information
print(data.head())

# column insights
print(data.info())

# check for any null values
print(data.isnull().sum())

#
#
# use the haversine formula to calculate the distance between two locations based on their latitudes and longitudes
# https://en.wikipedia.org/wiki/Haversine_formula
#
#

# set the earth's radius (in kilometers)
R = 6371


# convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)


# function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# calculate the distance between each pair of points
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'],
                                            data.loc[i, 'Restaurant_longitude'],
                                            data.loc[i, 'Delivery_location_latitude'],
                                            data.loc[i, 'Delivery_location_longitude'])

print(data.head())

#
#
# Exploratory Data Analysis
#
#


# relationship between the distance and time taken to deliver the food
figure = px.scatter(data_frame=data,
                    x="distance",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    trendline="ols",
                    title="Relationship between Distance and Time taken")
figure.show()
# conclusion: most delivery partners deliver food within 25-30 minutes, regardless of distance


# relationship between the time taken to deliver the food and the age of the delivery partner
figure = px.scatter(data_frame = data,
                    x="Delivery_person_Age",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    color = "distance",
                    trendline="ols",
                    title = "Relationship Between Time Taken and Age")
figure.show()
# conclusion: young delivery partners take less time to deliver the food compared to the elder partners


# relationship between the time taken to deliver the food and the ratings of the delivery partner
figure = px.scatter(data_frame = data,
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    color = "distance",
                    trendline="ols",
                    title = "Relationship Between Time Taken and Ratings")
figure.show()
# conclusion - delivery partners with higher ratings take less time to deliver the food compared to partners with low ratings


# type of food ordered by the customer and the type of vehicle used by the delivery partner affects the delivery time or not
fig = px.box(data,
             x="Type_of_vehicle",
             y="Time_taken(min)",
             color="Type_of_order")
fig.show()
# conclusion - not much difference between the time taken by delivery partners depending on the vehicle they are driving and the type of food they are delivering


# So the features that contribute most to the food delivery time based on the analysis are:
# (1) age of the delivery partner
# (2) ratings of the delivery partner
# (3) distance between the restaurant and the delivery location


#
# train ML model
#


# splitting the data
x = np.array(data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]])
y = np.array(data[["Time_taken(min)"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=9)

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))

