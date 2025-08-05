import numpy as np
import pandas as pd
import optuna
import joblib

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
joblib.dump(scaler_model, 'scaler.txt')
joblib.dump(scaler_model, 'scaler.save')

def objective(trial):   # Define optuna objective function
    # Hyperparameters for CatBoostRegressor
    catboost_iterations = trial.suggest_int('catboost_iterations', 50, 3000)
    catboost_depth = trial.suggest_int('catboost_depth', 3, 15)
    catboost_learning_rate = trial.suggest_float('catboost_learning_rate', 0.001, 0.3)
    catboost_l2_leaf_reg = trial.suggest_float('catboost_l2_leaf_reg', 1, 100, log=True)
    # catboost_bagging_temperature = trial.suggest_float('catboost_bagging_temperature', 0.1, 20.0, log=True),
    catbooost_random_strength = trial.suggest_float('catboost_random_strength', 1.0, 2.0)  
    
    catboost_regressor = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth,
                                           l2_leaf_reg=catboost_l2_leaf_reg,
                                           random_strength=catbooost_random_strength,
                                           learning_rate=catboost_learning_rate, random_state=42, verbose=0)
    

   
    catboost_regressor.fit(X_train, y_train)
    y_pred= catboost_regressor.predict(X_val)
    rmse_catboost= mean_squared_log_error(y_val, y_pred, squared=False)

    return rmse_catboost

# Hyperparameter tuning
study= optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500, show_progress_bar=True)
best_params= study.best_params
# Output
df_trials = study.trials_dataframe()
df_trials.to_excel('trials_catboost.xlsx')
df_best_params=pd.DataFrame.from_dict(best_params, orient='index').reset_index()
df_best_params.to_excel('best_params_catboost.xlsx')
