import pandas as pd

def preprocess_data(data):
    
    processed_data = data.copy()

    # Filtering based on charges
    Q1 = processed_data['charges'].quantile(0.25)
    Q3 = processed_data['charges'].quantile(0.75)
    IQR = Q3 - Q1
    processed_data = processed_data[(processed_data['charges'] >= Q1 - 1.5*IQR) & (processed_data['charges'] <= Q3 + 1.5*IQR)]

    # Handling missing values
    processed_data['medical_history'].fillna('Never', inplace=True)
    processed_data['family_medical_history'].fillna('Never', inplace=True)

    # Mapping categorical variables
    gender_mapping = {'female': 0, 'male': 1}
    smoker_mapping = {'no': 0, 'yes': 1}
    region_mapping = {'northeast': 0, 'southwest': 1, 'northwest': 2, 'southeast': 3}
    medical_history_mapping = {'Heart disease': 0, 'High blood pressure': 1, 'Diabetes': 2, 'Never': 3}
    family_medical_history_mapping = {'Heart disease': 0, 'High blood pressure': 1, 'Diabetes': 2, 'Never': 3}
    exercise_frequency_mapping = {'Rarely': 0, 'Occasionally': 1, 'Frequently': 2, 'Never': 3}
    occupation_mapping = {'Unemployed': 0, 'Student': 1, 'Blue collar': 2, 'White collar': 3}
    coverage_level_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2}

    processed_data['gender'] = processed_data['gender'].map(gender_mapping)
    processed_data['smoker'] = processed_data['smoker'].map(smoker_mapping)
    processed_data['region'] = processed_data['region'].map(region_mapping)
    processed_data['medical_history'] = processed_data['medical_history'].map(medical_history_mapping)
    processed_data['family_medical_history'] = processed_data['family_medical_history'].map(family_medical_history_mapping)
    processed_data['exercise_frequency'] = processed_data['exercise_frequency'].map(exercise_frequency_mapping)
    processed_data['occupation'] = processed_data['occupation'].map(occupation_mapping)
    processed_data['coverage_level'] = processed_data['coverage_level'].map(coverage_level_mapping)

    return processed_data