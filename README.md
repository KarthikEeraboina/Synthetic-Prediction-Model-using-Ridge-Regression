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
🧑‍💻 Author
E. Karthik Yadav
3rd-year student specializing in AI, ML, and Python-based software development.
