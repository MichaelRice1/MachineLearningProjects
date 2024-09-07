import pandas as pd
import numpy as np

#linear regression training
from sklearn.metrics import mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor 
from sklearn.linear_model import LinearRegression
  


player_data = pd.read_csv('player_data.csv')

#remove the points, rebounds and assists columns
X = player_data.drop(columns=['PTS','TRB','AST'])
Y = player_data[['PTS','TRB','AST']]

#split by year, 1999/2000 to 2022/2023 for training and 2023/2024 for testing
X_train = X[X['season'] != '2024']
X_test = X[X['season'] == '2024']
y_train = Y[X['season'] != '2024']
y_test = Y[X['season'] == '2024']



    

def linear_regression(X_train, y_train, X_test, y_test):
    
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, mae

def tabnet_regression(X_train, y_train, X_test, y_test):
    model = TabNetRegressor()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    
    return model, mae


if __name__ == "main":

    linear_regression(X_train, y_train, X_test, y_test)

    # tabnet_regression(X_train, y_train, X_test, y_test)
