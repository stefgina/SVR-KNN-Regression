#Description: This program predicts the price of FB stock for a specific day
#             using several Regression Algorithms.

#import the packages
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

#Load the data
df = pd.read_csv('FB_MONTHLY_STOCKS.csv')
df.head(7)

#Create the lists / X and y data set
dates = []
prices = []

#Get the number of rows and columns in the data set
df.shape

#The last row of data (this will be the sample that we test on)
df.tail(1)

#Get all of the data except for the last row
df = df.head(len(df)-1)
#print(df.shape)

df_dates = df.loc[:,'Date'] # Get all of the rows from the Date column
df_open = df.loc[:,'Open'] #Get all of the rows from the Open column

#Create the independent data set 'X' as dates
for date in df_dates:
  dates.append( [int(date.split('-')[2])] )
  
#Create the dependent data set 'y' as prices
for open_price in df_open:
  prices.append(float(open_price))

#See what days were recoreded in the data set
print("  ")
print("The days recorded for the train set shown bellow :")
print(dates)
print("  ")

#Function to make predictions using several regression algorithms
def predict_prices(dates, prices, x):

  print("  ")
  
  #Create the Regression Models
  svr_lin = SVR(kernel='linear', C=1e3)
  svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma=0.1)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  kneighbors = KNeighborsRegressor(n_neighbors=5)
  
  #Train the models on the dates and prices
  
  start_time1 = time.time()
  svr_lin.fit(dates,prices)
  linear_train_time = (time.time() - start_time1)

  start_time2 = time.time()
  svr_poly.fit(dates, prices)
  poly_train_time = (time.time() - start_time2)

  start_time3 = time.time()
  svr_rbf.fit(dates, prices)
  rbf_training_time = (time.time() - start_time3)

  start_time4 = time.time()
  kneighbors.fit(dates,prices)
  kneighbors_train_time = (time.time() - start_time4)
  
  #Plot the models on a graph to see which has the best fit
  plt.scatter(dates, prices, color = 'black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color = 'red', label='RBF model')
  plt.plot(dates, svr_lin.predict(dates), color = 'green', label='Linear model')
  plt.plot(dates, svr_poly.predict(dates), color = 'blue', label='Polynomial model')
  plt.plot(dates, kneighbors.predict(dates), color = 'yellow', label='K-Neighbors')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.title('Regression Models')
  plt.legend()
  plt.show()
  
  #return all model predictions
  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0], r2_score(prices,svr_rbf.predict(dates)), r2_score(prices,svr_lin.predict(dates)), r2_score(prices,svr_poly.predict(dates)), linear_train_time, poly_train_time, rbf_training_time, kneighbors.predict(x)[0], r2_score(prices,kneighbors.predict(dates)), kneighbors_train_time





#Predict the stock price of FB on day 31
predicted_price = predict_prices(dates, prices, [[31]])

print("   ")

# For RBF
print("---------------------------- FOR RBF ----------------------------")
print("RBF Training time was %s seconds " % predicted_price[8])
print("Test Score on 31/5/2019 is : " ,100- (abs(predicted_price[0]-180.279999)/180.279999)*100)
print("R^2 Score on whole Train set is : ", predicted_price[3])
print("Predicted Price is : ", predicted_price[0], "Real Price is 180.279999")
print("   ")

# For POLYNOMIAL
print("---------------------------- FOR POLYNOMIAL ----------------------------")
print("POLYNOMIAL Training time was %s seconds " % predicted_price[7])
print("Test Score on 31/5/2019 is : " ,100- (abs(predicted_price[1]-180.279999)/180.279999)*100)
print("R^2 Score on whole Train set is : ", predicted_price[4])
print("Predicted Price is : ", predicted_price[1], "Real Price is 180.279999")
print("   ")

# For LINEAR
print("---------------------------- FOR LINEAR ----------------------------")
print("LINEAR Training time was %s seconds " % predicted_price[6])
print("Test Score for 31day is : " ,100- (abs(predicted_price[2]-180.279999)/180.279999)*100)
print("R^2 Score on whole Train set is : ", predicted_price[5])
print("Predicted Price is : ", predicted_price[2], "Real Price is 180.279999")
print("   ")

# For FOR K-NEIGHBORS
print("---------------------------- FOR K-NEIGHBORS (n=5) ----------------------------")
print("K-NEIGHBORS Training time was %s seconds " % predicted_price[11])
print("Test Score for 31day is : " ,100- (abs(predicted_price[9]-180.279999)/180.279999)*100)
print("R^2 Score on whole Train set is : ", predicted_price[10])
print("Predicted Price is : ", predicted_price[9], "Real Price is 180.279999")
print("   ")






