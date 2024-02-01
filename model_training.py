import xgboost as xgb
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
import logging

data = pd.read_csv("insurance_dataset.csv")

clean_data = preprocess_data(data)

X = clean_data.drop('charges', axis=1)

y = clean_data['charges']

def train_xgboost_model(X, y):
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 7,
        'seed': 42
    }

   
    xgb_model = xgb.XGBRegressor(**params)

    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    
    y_test_pred = xgb_model.predict(X_test)

   
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    logging.info(f"R2 Score (test): {r2_test}")
    logging.info(f"Mean Absolute Error (test): {mae_test}")

    
    joblib.dump(xgb_model, 'insurance_model.pkl')

    return xgb_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trained_model = train_xgboost_model(X, y)
