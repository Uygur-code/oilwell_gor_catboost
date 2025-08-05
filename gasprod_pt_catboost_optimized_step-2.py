import numpy as np
import pandas as pd
# import optuna
import joblib

import plotly.graph_objects as go
from plotly._subplots import make_subplots

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import  r2_score, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import RobustScaler

df_input= pd.read_excel('GasProd_PT.xlsx', sheet_name='pt_input')
X= df_input[['WHFP', 'UP_PR','OIL_RATE']]
y= df_input['GOR']
X_train_unscaled, X_val_unscaled, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
scaler_model = scaler.fit(X_train_unscaled)
X_train = scaler_model.transform(X_train_unscaled)
X_val = scaler_model.transform(X_val_unscaled)
# joblib.dump(scaler_model, 'scaler.txt')
# joblib.dump(scaler_model, 'scaler.save')

df_x = pd.read_excel('trials_catboost.xlsx', sheet_name='Sheet2')
j = 0


trained_catboost_regressor = [None] * 500
test_catboost_regressor = [None] * 500
rmse_catboost = [None] * 500
trained_rmse_catboost =[None] * 500

y_trained_pred = pd.DataFrame()
y_test_pred = pd.DataFrame()
df_trained_pred = pd.DataFrame({'M_GOR' : y_train})
df_test_pred = pd.DataFrame({'M_GOR' : y_val})
fig = make_subplots(rows=2, cols=1)
ideal_trace = go.Scatter(x=[0,7000], y=[0,7000], 
                         name='Ideal Trace',mode='lines',
                         line_dash='dash', line_color='#615e5e', line_width=1)

# ctb_i = [2, 7, 9, 39, 46, 49, 95, 108]
ctb_i = range(500)
for i in range (500):
 df_y = df_x[['number',i]]
 best_params = dict(df_y[:].values)
    
 best_catboost_regressor = CatBoostRegressor(iterations=best_params['catboost_iterations'],
                                       depth=best_params['catboost_depth'],
                                       l2_leaf_reg=best_params['catboost_l2_leaf_reg'],
                                       random_strength=best_params['catboost_random_strength'],
                                       learning_rate=best_params['catboost_learning_rate'],
                                       random_state=42, verbose=0)

  
 best_catboost_regressor.fit(X_train, y_train)
  
 trained_catboost_regressor[i] = best_catboost_regressor.predict(X_train)
 
 trained_rmse_catboost[i]= mean_squared_log_error(y_train, trained_catboost_regressor[i], squared=False)
 
 # Use test data to predict
 
 test_catboost_regressor[i] = best_catboost_regressor.predict(X_val)
 
 rmse_catboost[i]= mean_squared_log_error(y_val, test_catboost_regressor[i], squared=False)

 
 # print(f'Trained Catboost RMSE ({j}): {trained_rmse_catboost[j]}')
 
 #  print(f'Catboost RMSE ({j}): {rmse_catboost[j]}')
 
 # plot_titles = ()

 fig.add_trace(go.Scatter(x=y_train, y=trained_catboost_regressor[i],
              mode='markers', name='trained_catboost_regressor', marker_color='#125b96'), row=1, col=1)

 fig.add_trace(go.Scatter(x=y_val, y=test_catboost_regressor[i],
              mode='markers', name='test_catboost_regressor', marker_color='#EF553B'), row=1, col=1)

 fig.add_trace( ideal_trace, row=1, col=1 )

 fig.add_trace(go.Scatter(x=y_train, y=trained_catboost_regressor[i],
              mode='markers', name='trained_catboost_regressor', marker_color='#125b96'), row=2, col=1)

 fig.add_trace(go.Scatter(x=y_val, y=test_catboost_regressor[i],
              mode='markers', name='test_catboost_regressor', marker_color='#EF553B'), row=2, col=1)

 fig.add_trace( ideal_trace, row=2, col=1 )
 
 j=j+1


 fig.update_xaxes(range=[0,7000], gridcolor='#f3f6f4',title='Measured GOR (SCF/STB)', mirror=True, linecolor='black', row=1, col=1)
 fig.update_yaxes(range=[0,7000], gridcolor='#f3f6f4', title='Calculated GOR (SCF/STB)', mirror=True, linecolor='black', row=1, col=1)
 fig.update_xaxes(range=[0,1000], gridcolor='#f3f6f4',title='Measured GOR (SCF/STB)', mirror=True, linecolor='black', row=2, col=1)
 fig.update_yaxes(range=[0,1000], gridcolor='#f3f6f4', title='Calculated GOR (SCF/STB)', mirror=True, linecolor='black', row=2, col=1)


 fig.update_layout(showlegend=False,height=1000, width=1000,
                   title_text='Catboost Regression :' + str(i) + '  // Train RMSLE :' + str(trained_rmse_catboost[i]) + ' /  Test RMSLE :' + str(rmse_catboost[i]),
                   plot_bgcolor = '#ffffff', font=dict(family="Times New Roman",size=14, color='black'))
 fig.write_image('D:/GasProd_PT/images_catboost/figure_' + str(i) + '_' + str(best_params['value'])+ '.png')
 fig.data = []
 y_trained_pred['Trnd' + str(i)] = trained_catboost_regressor[i]
 y_test_pred['Test' + str(i)] = test_catboost_regressor[i]



df_trained_pred = pd.concat([df_trained_pred, y_trained_pred], axis=1)
df_test_pred = pd.concat([df_test_pred, y_test_pred], axis=1)

df_trained_pred.to_excel('Catboost_trained_pred.xlsx')
df_test_pred.to_excel('Catboost_test_pred.xlsx')
df_rmsle = pd.DataFrame({'Item' :ctb_i,'Train_RMSLE' : trained_rmse_catboost, 'Test_RMSLE' : rmse_catboost})
df_rmsle.to_excel('Catboost_RMSLE.xlsx')