import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class SensorModel:
    def __init__(self):
        pass

    def train(self):
        # Load the data
        data = pd.read_csv('NPK_dataset.csv')

        # Encode the categorical label (crop type)
        le = LabelEncoder()
        data['label'] = le.fit_transform(data['label'])

        # Split the data into features (X) and targets (y for N, P, K)
        X = data[['temperature', 'humidity', 'ph', 'rainfall', 'label']]

        # Independent target variables
        y_N = data['N']
        y_P = data['P']
        y_K = data['K']

        # Split into train and test sets
        X_train, X_test, y_N_train, y_N_test = train_test_split(X, y_N, test_size=0.2, random_state=42)
        X_train, X_test, y_P_train, y_P_test = train_test_split(X, y_P, test_size=0.2, random_state=42)
        X_train, X_test, y_K_train, y_K_test = train_test_split(X, y_K, test_size=0.2, random_state=42)

        # Define and train separate XGBoost models for N, P, and K
        model_N = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
        model_P = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
        model_K = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)

        model_N.fit(X_train, y_N_train)
        model_P.fit(X_train, y_P_train)
        model_K.fit(X_train, y_K_train)

        # Save the models
        model_N.save_model('N_model.json')
        model_P.save_model('P_model.json')
        model_K.save_model('K_model.json')

        # Plot
        y_N_pred = model_N.predict(X_test)
        y_P_pred = model_P.predict(X_test)
        y_K_pred = model_K.predict(X_test)

        print("RMSE for N: ", np.sqrt(mean_squared_error(y_N_test, y_N_pred)))
        print("RMSE for P: ", np.sqrt(mean_squared_error(y_P_test, y_P_pred)))
        print("RMSE for K: ", np.sqrt(mean_squared_error(y_K_test, y_K_pred)))

        # Plot separate - too dense to show on the same plot

        plt.figure(figsize=(12, 6))
        plt.plot(y_N_test.values, label='Actual N')
        plt.plot(y_N_pred, label='Predicted N', linestyle='dashed')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Model Predictions vs Actual Values')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(y_P_test.values, label='Actual P')
        plt.plot(y_P_pred, label='Predicted P', linestyle='dashed')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Model Predictions vs Actual Values')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(y_K_test.values, label='Actual K')
        plt.plot(y_K_pred, label='Predicted K', linestyle='dashed')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Model Predictions vs Actual Values')
        plt.legend()
        plt.show()

    def predict(self, temperature, humidity, ph, rainfall, crop):
        if (crop not in ['rice', 'wheat', 'maize', 'lentil', 'jute', 'coffee', 'cotton', 'sugarcane', 'tobacco', 'pepper', 'apple', 'banana', 'mango', 'grapes']):
            return 'Invalid crop type'

        # Load the model
        N_model = xgb.XGBRegressor()
        P_model = xgb.XGBRegressor()
        K_model = xgb.XGBRegressor()

        N_model.load_model('N_model.json')
        P_model.load_model('P_model.json')
        K_model.load_model('K_model.json')

        # Load the data again
        data = pd.read_csv('NPK_dataset.csv')

        # Encode the crop type
        le = LabelEncoder()
        data['label'] = le.fit_transform(data['label'])

        # Predict the N, P, and K values
        X = data[['temperature', 'humidity', 'ph', 'rainfall', 'label']]
        N = N_model.predict(X)
        P = P_model.predict(X)
        K = K_model.predict(X)

        prediction = {
            'N': N[0],
            'P': P[0],
            'K': K[0]
        }

        print(prediction)

        return prediction
    
if __name__ == '__main__':
    model = SensorModel()
    model.train()
    model.predict(25, 70, 6.5, 200, 'rice')