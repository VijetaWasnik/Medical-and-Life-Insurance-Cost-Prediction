<h1>Health and Life Insurance Premium Prediction</h1>
<p>This project aims to develop a predictive model for estimating health insurance premiums in the United States. Accurately predicting premiums is crucial for insurance companies to optimize pricing strategies and ensure customer satisfaction.</p>

<h3>Dataset</h3>
The dataset used in this project contains 1 million records with information on various factors influencing health insurance premiums, including age, gender, BMI, number of children, smoking status, region, income, education, occupation, and type of insurance plan. The dataset is well-prepared, balanced, and free from null values and duplicates.
<br>

<h3>Methodology</h3>
<strong>1. Data Splitting:</strong> The dataset was split into training (70%), validation (20%), and test (10%) sets using train_test_split from scikit-learn.
<br>
<strong>2. Model Selection:</strong> A Random Forest Regressor model was chosen due to its ability to handle non-linear relationships and high-dimensional data effectively.
<br>
<strong>3. Model Training:</strong> The Random Forest Regressor was trained on the training set with pre-defined parameters (n_estimators=100, max_depth=7, random_state=42).
<br>
<strong>4. Model Evaluation:</strong> The model's performance was evaluated on the validation set using metrics such as R-squared score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
<br>
<strong>5. Hyperparameter Tuning:</strong> Hyperparameters (n_estimators and max_depth) were tuned using GridSearchCV to minimize the Mean Absolute Error (MAE) on the training set.
<br>
<strong>6. Model Interpretation:</strong> SHAP (SHapley Additive exPlanations) values were used to interpret the model's predictions and understand feature importance.
<br>

<h3>Results</h3>
The R-squared score for the validation set was approximately 0.91, indicating a good fit of the model to the data.
<br>
The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the validation set were around 1063 and 1318, respectively.
<br>

<h3>Usage</h3>
<strong>model_training.py:</strong>Python script for training the predictive model.
<br>
<strong>prediction.py:</strong>Python script for making predictions using the trained model.
<br>
Ensure that the dataset goes through the same preprocessing steps during model building and prediction phases.
<br>

<h3>Dependencies</h3>
Python 3.x
<br>
scikit-learn
<br>
numpy
<br>
shap
<br>

<h3>Future Improvements</h3>
Explore additional feature engineering techniques to enhance model performance.
<br>
Experiment with other machine learning algorithms to compare performance.
<br>
Incorporate additional data sources to improve prediction accuracy.
<br>

<h3>Contributing</h3>
Contributions to the project are welcome! Feel free to submit bug fixes, feature requests, or improvements via pull requests.
<br>

<h3>Contact</h3>
For inquiries or collaboration, contact wasnik_vijeta@gmail.com .
