import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import __version__ as skl_version

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------
# STEP 1: Load data directly from text files
# ---------------------------------------------------------------------
TRAIN_PATH = "codetest_train.txt"
TEST_PATH = "codetest_test.txt"

train = pd.read_csv(TRAIN_PATH, sep='\t', header=0)
test = pd.read_csv(TEST_PATH, sep='\t', header=0)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ---------------------------------------------------------------------
# STEP 2: Split features and target
# ---------------------------------------------------------------------
if 'target' in train.columns:
    y = train['target'].values
    X = train.drop(columns=['target'])
else:
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1].values

X_test = test.copy()

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", cat_cols)
print(f"Numeric features: {len(num_cols)}")

# ---------------------------------------------------------------------
# STEP 3: Define preprocessing + model pipeline (version-safe)
# ---------------------------------------------------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Handle sklearn 1.4+ argument change
if tuple(map(int, skl_version.split('.')[:2])) >= (1, 4):
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
else:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
    ])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 10),
                      scoring='neg_mean_squared_error',
                      cv=3))
])

# ---------------------------------------------------------------------
# STEP 4: Validation
# ---------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_tr, y_tr)
val_preds = pipeline.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)

print(f"\nValidation MSE: {val_mse:.6f}")
print(f"Validation RMSE: {np.sqrt(val_mse):.6f}")

# ---------------------------------------------------------------------
# STEP 5: Plot Predicted vs Actual
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_val, val_preds, alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual (Validation Set)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# STEP 6: Train final model and predict test set
# ---------------------------------------------------------------------
pipeline.fit(X, y)
preds = pipeline.predict(X_test)

# Save predictions (one per line)
preds_path = "predictions.txt"
np.savetxt(preds_path, preds, fmt="%.6f")

# Save trained model
model_path = "ridge_pipeline.joblib"
joblib.dump(pipeline, model_path)

print(f"\n✅ Predictions saved to: {preds_path}")
print(f"✅ Model saved to: {model_path}")

# ---------------------------------------------------------------------
# STEP 7: Feature Importance Visualization
# ---------------------------------------------------------------------
try:
    ohe = pipeline.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
    ohe_feature_names = []
    if cat_cols:
        cat_categories = ohe.categories_
        for col, cats in zip(cat_cols, cat_categories):
            for c in cats[1:]:  # drop first because drop='first'
                ohe_feature_names.append(f"{col}__{c}")

    feature_names = num_cols + ohe_feature_names
    coef = pipeline.named_steps['ridge'].coef_
    coef_imp = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)

    print("\nTop 10 most important features:")
    print(coef_imp.head(10).to_string())

    # Plot top 20 feature importances
    plt.figure(figsize=(10, 6))
    coef_imp.head(20).sort_values().plot(kind='barh', color='teal')
    plt.title("Top 20 Most Important Features (Absolute Coefficients)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\nFeature importance not available:", e)
