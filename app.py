from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_insurance', methods=['POST'])

def predict_insurance():
    print(request.method)
    
    insurance_model = joblib.load('insurance_model.pkl')

    input_data = {
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'bmi': float(request.form['bmi']),
        'children': int(request.form['children']),
        'smoker': request.form['smoker'],
        'region': request.form['region'],
        'medical_history': request.form['medical_history'],
        'family_medical_history': request.form['family_medical_history'],
        'exercise_frequency': request.form['exercise_frequency'],
        'occupation': request.form['occupation'],
        'coverage_level': request.form['coverage_level']
    }


    input_data['gender'] = 1 if input_data['gender'] == 'male' else 0
    input_data['smoker'] = 1 if input_data['smoker'] == 'yes' else 0
    if input_data['region'] == 'northeast':
        input_data['region'] = 0
    elif input_data['region'] == 'southwest':
        input_data['region'] = 1
    elif input_data['region'] == 'northwest':
        input_data['region'] = 2
    else:
        input_data['region'] = 3

    if input_data['medical_history'] == 'Heart disease':
        input_data['medical_history'] = 0
    elif input_data['medical_history'] == 'High blood pressure':
        input_data['medical_history'] = 1
    elif input_data['medical_history'] == 'Diabetes':
        input_data['medical_history'] = 2
    else:
        input_data['medical_history'] = 3

    if input_data['family_medical_history'] == 'Heart disease':
        input_data['family_medical_history'] = 0
    elif input_data['family_medical_history'] == 'High blood pressure':
        input_data['family_medical_history'] = 1
    elif input_data['family_medical_history'] == 'Diabetes':
        input_data['family_medical_history'] = 2
    else:
        input_data['family_medical_history'] = 3

    if input_data['exercise_frequency'] == 'Rarely':
        input_data['exercise_frequency'] = 0
    elif input_data['exercise_frequency'] == 'Occasionally':
        input_data['exercise_frequency'] = 1
    elif input_data['exercise_frequency'] == 'Frequently':
        input_data['exercise_frequency'] = 2
    else:
        input_data['exercise_frequency'] = 3

    if input_data['occupation'] == 'Unemployed':
        input_data['occupation'] = 0
    elif input_data['occupation'] == 'Student':
        input_data['occupation'] = 1
    elif input_data['occupation'] == 'Blue collar':
        input_data['occupation'] = 2
    else:
        input_data['occupation'] = 3


    if input_data['coverage_level'] == 'Basic':
        input_data['coverage_level'] = 0
    elif input_data['coverage_level'] == 'Standard':
        input_data['coverage_level'] = 1
    else:
        input_data['coverage_level'] = 2


    df = pd.DataFrame([input_data])
    
    # Make predictions
    insurance_prediction = insurance_model.predict(df[['age','gender','bmi','children','smoker','region','medical_history','family_medical_history','exercise_frequency','occupation','coverage_level']])[0]
    
    print(insurance_prediction)
 

    return render_template('result.html', insurance_result=insurance_prediction)

if __name__ == '__main__':
    app.run(debug=True)