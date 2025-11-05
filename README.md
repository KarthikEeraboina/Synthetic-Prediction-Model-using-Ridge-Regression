# Synthetic-Prediction-Model-using-Ridge-Regression

🧠 Predictive Modeling Challenge
Synthetic Dataset Regression Task
This repository contains my solution for a predictive modeling challenge using synthetic datasets generated from the same underlying data model.
The goal is to build a regression model using the training dataset and predict the withheld target values for the test dataset.
________________________________________
📂 Dataset Description
Two tab-delimited files are provided:
File	Description	Shape
codetest_train.txt	Training data containing 5,000 records × 254 features + 1 target	5,000 × 255
codetest_test.txt	Test data containing 1,000 records × 254 features (no target)	1,000 × 254
Objective:
Predict the target variable for all 1,000 rows in the test dataset and evaluate accuracy using Mean Squared Error (MSE).
________________________________________
⚙️ Approach
1. Data Preprocessing
•	Loaded the train and test files using Pandas.
•	Identified numeric vs categorical columns automatically.
•	Applied median imputation for missing numeric values.
•	Encoded categorical features using Label Encoding (combined mapping from train + test).
•	Ensured both train and test had perfectly aligned feature sets.
2. Model Training
Two model options were implemented:
•	Primary: LightGBMRegressor
o	Gradient Boosting Decision Tree model.
o	Parameters: n_estimators=500, learning_rate=0.05, num_leaves=31.
o	5-Fold Cross Validation (OOF RMSE calculated).
•	Fallback: RidgeCV
o	Linear regression with cross-validated L2 regularization.
o	Used if LightGBM is unavailable (fast baseline).
3. Evaluation
•	Metric: Root Mean Squared Error (RMSE) on out-of-fold predictions.
•	RMSE gives an estimate of the average prediction error magnitude.
•	Model performance visualized using a Predicted vs Actual scatter plot.
________________________________________
📈 Results
•	Out-of-Fold RMSE: ≈ (fill after training)
•	Visual Evaluation:
Scatter plot (pred_vs_actual.png) showing predicted values against actual target values.
Points close to the red dashed line indicate accurate predictions.
<p align="center"> <img src="pred_vs_actual.png" alt="Predicted vs Actual" width="500"/> </p> 
________________________________________
📊 Output Files
File	Description
predictions.txt	1,000 predicted target values (one per line)
pred_vs_actual.png	Scatter plot comparing predicted vs actual
model_artifact.pkl	Pickled trained model + encoders
writeup.txt	Summary of model performance (RMSE, algorithm, etc.)
________________________________________
🧩 How to Run
Step 1. Clone the repository
git clone https://github.com/yourusername/predictive-model-synthetic.git
cd predictive-model-synthetic
Step 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib
Step 3. Run training
Make sure both dataset files are in the same folder, then run:
python train_and_predict_with_graph.py
Step 4. View outputs
After running:
•	predictions.txt → Predicted values for test set
•	pred_vs_actual.png → Model performance graph
•	writeup.txt → Summary report
________________________________________
🧠 Key Learnings
•	Label encoding and proper imputation are critical for consistency between train and test sets.
•	Cross-validation helps estimate generalization error reliably.
•	Visualization (Predicted vs Actual) quickly reveals overfitting or underfitting.
________________________________________
🚀 Future Improvements
•	Hyperparameter tuning for LightGBM using Optuna or GridSearch.
•	Feature selection / dimensionality reduction (e.g., PCA).
•	Ensembling multiple models (LightGBM + XGBoost + Ridge).
•	Deploying model as an API endpoint using Streamlit or Flask.
________________________________________
🧑‍💻 Author
E. Karthik Yadav
2nd-year student specializing in AI, ML, and Python-based software development.

Visual Evaluation:
Scatter plot (pred_vs_actual.png) showing predicted values against actual target values.
Points close to the red dashed line indicate accurate predictions.

<p align="center"> <img src="pred_vs_actual.png" alt="Predicted vs Actual" width="500"/> </p>
📊 Output Files
File	Description
predictions.txt	1,000 predicted target values (one per line)
pred_vs_actual.png	Scatter plot comparing predicted vs actual
model_artifact.pkl	Pickled trained model + encoders
writeup.txt	Summary of model performance (RMSE, algorithm, etc.)
🧩 How to Run
Step 1. Clone the repository
git clone https://github.com/yourusername/predictive-model-synthetic.git
cd predictive-model-synthetic

Step 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib

Step 3. Run training

Make sure both dataset files are in the same folder, then run:

python train_and_predict_with_graph.py

Step 4. View outputs

After running:

predictions.txt → Predicted values for test set

pred_vs_actual.png → Model performance graph

writeup.txt → Summary report

🧠 Key Learnings

Label encoding and proper imputation are critical for consistency between train and test sets.

Cross-validation helps estimate generalization error reliably.

Visualization (Predicted vs Actual) quickly reveals overfitting or underfitting.

🚀 Future Improvements

Hyperparameter tuning for LightGBM using Optuna or GridSearch.

Feature selection / dimensionality reduction (e.g., PCA).

Ensembling multiple models (LightGBM + XGBoost + Ridge).

Deploying model as an API endpoint using Streamlit or Flask.

🧑‍💻 Author

E. Karthik Yadav
2nd-year student specializing in AI, ML, and Python-based software development.🧠 Predictive Modeling Challenge
Synthetic Dataset Regression Task

This repository contains my solution for a predictive modeling challenge using synthetic datasets generated from the same underlying data model.
The goal is to build a regression model using the training dataset and predict the withheld target values for the test dataset.

📂 Dataset Description

Two tab-delimited files are provided:

File	Description	Shape
codetest_train.txt	Training data containing 5,000 records × 254 features + 1 target	5,000 × 255
codetest_test.txt	Test data containing 1,000 records × 254 features (no target)	1,000 × 254

Objective:
Predict the target variable for all 1,000 rows in the test dataset and evaluate accuracy using Mean Squared Error (MSE).

⚙️ Approach
1. Data Preprocessing

Loaded the train and test files using Pandas.

Identified numeric vs categorical columns automatically.

Applied median imputation for missing numeric values.

Encoded categorical features using Label Encoding (combined mapping from train + test).

Ensured both train and test had perfectly aligned feature sets.

2. Model Training

Two model options were implemented:

Primary: LightGBMRegressor

Gradient Boosting Decision Tree model.

Parameters: n_estimators=500, learning_rate=0.05, num_leaves=31.

5-Fold Cross Validation (OOF RMSE calculated).

Fallback: RidgeCV

Linear regression with cross-validated L2 regularization.

Used if LightGBM is unavailable (fast baseline).

3. Evaluation

Metric: Root Mean Squared Error (RMSE) on out-of-fold predictions.

RMSE gives an estimate of the average prediction error magnitude.

Model performance visualized using a Predicted vs Actual scatter plot.

📈 Results

Out-of-Fold RMSE: ≈ (fill after training)

Visual Evaluation:
Scatter plot (pred_vs_actual.png) showing predicted values against actual target values.
Points close to the red dashed line indicate accurate predictions.

<p align="center"> <img src="pred_vs_actual.png" alt="Predicted vs Actual" width="500"/> </p>
📊 Output Files
File	Description
predictions.txt	1,000 predicted target values (one per line)
pred_vs_actual.png	Scatter plot comparing predicted vs actual
model_artifact.pkl	Pickled trained model + encoders
writeup.txt	Summary of model performance (RMSE, algorithm, etc.)
🧩 How to Run
Step 1. Clone the repository
git clone https://github.com/yourusername/predictive-model-synthetic.git
cd predictive-model-synthetic

Step 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib

Step 3. Run training

Make sure both dataset files are in the same folder, then run:

python train_and_predict_with_graph.py

Step 4. View outputs

After running:

predictions.txt → Predicted values for test set

pred_vs_actual.png → Model performance graph

writeup.txt → Summary report

🧠 Key Learnings

Label encoding and proper imputation are critical for consistency between train and test sets.

Cross-validation helps estimate generalization error reliably.

Visualization (Predicted vs Actual) quickly reveals overfitting or underfitting.

🚀 Future Improvements

Hyperparameter tuning for LightGBM using Optuna or GridSearch.

Feature selection / dimensionality reduction (e.g., PCA).

Ensembling multiple models (LightGBM + XGBoost + Ridge).

Deploying model as an API endpoint using Streamlit or Flask.

🧑‍💻 Author

E. Karthik Yadav
3rd-year student specializing in AI, ML, and Python-based software development.
