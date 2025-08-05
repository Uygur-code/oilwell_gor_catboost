import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import  r2_score, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import RobustScaler
import joblib

df_input= pd.read_excel('gor_pred_input.xlsx', sheet_name='Sheet1')
X= df_input[['WHFP', 'UP_PR','OIL_RATE']]

scaler_model = joblib.load('scaler.save')
x_pred = scaler_model.transform(X)

# Trial 67

catboost_reg = joblib.load('catboost_gor_model.save')

gor_pred = catboost_reg.predict(x_pred)


df_input['GOR'] = gor_pred

df_input.to_excel('gor_pred_output_2024-SEP.xlsx')


    





