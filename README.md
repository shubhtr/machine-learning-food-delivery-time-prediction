# Machine Learning Food Delivery Time Prediction
#### by Shubhrendu Tripathi

This project applies machine learning to predict food delivery times using Long Short-Term Memory (LSTM) network. 
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.

## Setup

* Python 3.10
* scikit-learn 1.3
* pandas 2.0
* numpy 1.25
* plotly 5.16

## Dataset

The dataset for this project originates from 
[Gaurav Malik's Kaggle dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset). 


### Features in the Dataset

* ID: order ID number 
* Delivery_person_ID: ID number of the delivery partner
* Delivery_person_Age: Age of the delivery partner
* Delivery_person_Ratings: ratings of the delivery partner based on past deliveries
* Restaurant_latitude: The latitude of the restaurant
* Restaurant_longitude: The longitude of the restaurant
* Delivery_location_latitude: The latitude of the delivery location
* Delivery_location_longitude: The longitude of the delivery location
* Type_of_order: The type of meal ordered by the customer
* Type_of_vehicle: The type of vehicle delivery partner rides
* Time_taken(min): The time taken by the delivery partner to complete the order
