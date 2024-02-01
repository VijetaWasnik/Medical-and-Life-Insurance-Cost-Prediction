import pandas as pd
import xgboost as xgb
import pickle
from data_preprocessing import preprocess_data  

def load_model(model_path='insurance_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_with_model(model, data):
    preprocessed_data = preprocess_data(data)
    X = preprocessed_data.drop('charges', axis=1)
    predictions = model.predict(X)
    return predictions

if __name__ == '__main__':

    xgb_model = load_model()

    input_data = pd.read_csv('insurance_dataset.csv')

    result = predict_with_model(xgb_model, input_data)

    print("Predictions:", result)
