import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import warnings
warnings.filterwarnings('ignore')
import joblib
import json

# =========================================================
# IMPROVED FEATURE ENGINEERING (NO LEAKAGE)
# =========================================================
def create_features_safe(df):
    """Create features without leakage - safe for train/test split"""
    df = df.copy()
    
    # Only create features that don't leak future/target information
    # Avoid using loan approval criteria directly
    df['income_to_existing_debt'] = df['annual_income'] / (df['existing_loan_balance'] + 1)
    df['credit_utilization'] = df['existing_loan_balance'] / (df['annual_income'] + 1)
    
    # Employment stability
    df['employment_stability'] = df['years_employed'] / (df['age'] - 18 + 1)
    
    # Credit behavior
    df['credit_inquiry_intensity'] = df['credit_inquiry_last_6m'] / (df['credit_age_months'] + 1)
    df['payment_stability'] = df['credit_score'] / (df['payment_history_default'] + 1)
    
    return df

# =========================================================
# DATA LOADING & PREPARATION
# =========================================================
print("="*60)
print("LOAN APPROVAL MODEL TRAINING - FIXED VERSION")
print("="*60)

df = pd.read_csv('loan_history.csv')

# Create target BEFORE any feature engineering
df['target'] = df['approval_status'].map({'Approved': 1, 'Rejected': 0})

# Initial split - BEFORE any feature engineering
X = df.drop(['target', 'approval_status', 'rejection_reason', 'approval_type', 'pan_number'], axis=1, errors='ignore')
y = df['target']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Now apply feature engineering SEPARATELY to train and test
X_train = create_features_safe(X_train_raw)
X_test = create_features_safe(X_test_raw)

print("\nDataset Information:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Approved in train: {y_train.sum()}")
print(f"Approved in test: {y_test.sum()}")

# =========================================================
# SELECT FEATURES (REDUCE DIMENSIONALITY)
# =========================================================
# Define numerical and categorical features AFTER feature engineering
numerical_features = [
    'age', 'years_employed', 'annual_income', 'monthly_income',
    'existing_loan_balance', 'existing_emi_monthly', 'credit_score', 'cibil_score',
    'payment_history_default', 'credit_inquiry_last_6m', 'num_open_accounts',
    'num_delinquent_accounts', 'property_value', 'requested_loan_amount',
    'requested_loan_tenure', 'pre_approved_limit', 'debt_to_income_ratio',
    'loan_to_income_ratio', 'credit_age_months', 'income_to_existing_debt',
    'credit_utilization', 'employment_stability', 'credit_inquiry_intensity',
    'payment_stability'
]

# Remove highly correlated features manually
numerical_features = [
    'age', 'years_employed', 'annual_income',  # Keep only annual_income, not monthly
    'existing_loan_balance', 'existing_emi_monthly', 'credit_score',
    'payment_history_default', 'credit_inquiry_last_6m', 'num_open_accounts',
    'property_value', 'requested_loan_amount', 'requested_loan_tenure',
    'debt_to_income_ratio', 'loan_to_income_ratio', 'credit_age_months',
    'income_to_existing_debt', 'credit_utilization', 'employment_stability'
]

categorical_features = [
    'gender', 'city', 'employment_type', 'education_level',
    'marital_status', 'home_ownership', 'property_type'
]

# =========================================================
# BUILD PIPELINE WITH REGULARIZATION & FEATURE SELECTION
# =========================================================
print("\n" + "="*60)
print("TRAINING MODEL WITH REGULARIZATION")
print("="*60)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# Feature selection step
feature_selector = VarianceThreshold(threshold=0.01)  # Remove low variance features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# STRONGER REGULARIZATION to prevent overfitting
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', feature_selector),
    ('classifier', LogisticRegression(
        C=0.01,  # Much stronger regularization
        penalty='l2', 
        solver='liblinear', 
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

# =========================================================
# CROSS-VALIDATION (PROPER VALIDATION)
# =========================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # More folds
logreg_scores = cross_val_score(logreg_pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

print(f"Cross-validation F1 scores: {logreg_scores}")
print(f"Mean CV F1: {logreg_scores.mean():.4f} (+/- {logreg_scores.std():.4f})")

# =========================================================
# FINAL TRAINING & EVALUATION
# =========================================================
logreg_pipeline.fit(X_train, y_train)

# Training performance
y_train_pred = logreg_pipeline.predict(X_train)
y_train_proba = logreg_pipeline.predict_proba(X_train)[:, 1]
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Test performance
y_test_pred = logreg_pipeline.predict(X_test)
y_test_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTraining F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
print(f"Test F1:     {test_f1:.4f}, AUC: {test_auc:.4f}")

# Check for overfitting
overfitting_gap = train_f1 - test_f1
print(f"\nOverfitting gap (Train F1 - Test F1): {overfitting_gap:.4f}")

if overfitting_gap > 0.05:
    print("⚠️  WARNING: Significant overfitting detected!")
elif overfitting_gap > 0.02:
    print("⚠️  Moderate overfitting detected")
else:
    print("✓ Model appears well-regularized")

# =========================================================
# FEATURE IMPORTANCE ANALYSIS
# =========================================================
# Get feature names after preprocessing
preprocessor.fit(X_train)
feature_names = []

# Numerical features
feature_names.extend(numerical_features)

# Categorical features
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
feature_names.extend(cat_features)

# Get coefficients (only for features that passed variance threshold)
model = logreg_pipeline.named_steps['classifier']
feature_selector = logreg_pipeline.named_steps['feature_selector']

# Get selected feature indices
selected_indices = feature_selector.get_support()
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]
selected_coefficients = model.coef_[0]

print(f"\nSelected {len(selected_features)} out of {len(feature_names)} features")
print(f"\nTop 10 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': selected_features,
    'coefficient': selected_coefficients,
    'abs_coefficient': np.abs(selected_coefficients)
}).sort_values('abs_coefficient', ascending=False)

print(importance_df.head(10))

# =========================================================
# SAVE MODEL
# =========================================================
best_model = logreg_pipeline
best_model_name = "Regularized Logistic Regression"

joblib.dump(best_model, 'underwriting_model_regularized.pkl')

model_info = {
    'model_name': best_model_name,
    'train_f1': float(train_f1),
    'test_f1': float(test_f1),
    'overfitting_gap': float(overfitting_gap),
    'cv_mean_f1': float(logreg_scores.mean()),
    'cv_std_f1': float(logreg_scores.std()),
    'regularization_strength': 0.01,
    'selected_features': selected_features[:20],  # Top 20 features
    'threshold': 0.5
}

with open('model_info_regularized.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*60)
print("MODEL SAVED SUCCESSFULLY")
print("="*60)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_loan_approval_safe(customer_data):
    """Safe prediction function that prevents leakage"""
    model = joblib.load('underwriting_model_regularized.pkl')
    customer_data = create_features_safe(customer_data)
    
    X_new = customer_data[numerical_features + categorical_features]
    
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'customer_id': customer_data.iloc[i].get('customer_id', i+1),
            'approval_status': 'Approved' if pred == 1 else 'Rejected',
            'approval_probability': float(prob),
            'confidence': 'High' if prob > 0.8 or prob < 0.2 else 'Medium',
            'recommendation': 'Approve' if pred == 1 else 'Reject',
            'risk_factors': []
        })
    
    return results

# =========================================================
# RECOMMENDATIONS TO FURTHER PREVENT OVERFITTING
# =========================================================
print("\n" + "="*60)
print("RECOMMENDATIONS FOR FUTURE IMPROVEMENT")
print("="*60)

if overfitting_gap > 0.03:
    print("1. Increase regularization strength (try C=0.001)")
    print("2. Add dropout-like regularization (if using neural networks)")
    print("3. Try ensemble methods (Random Forest, Gradient Boosting)")
    print("4. Collect more training data")
    print("5. Add more domain-specific features")
    print("6. Use feature importance to remove noisy features")

if test_f1 < 0.75:
    print("\n7. Consider different algorithms (XGBoost, LightGBM)")
    print("8. Feature engineering: create more meaningful features")
    print("9. Hyperparameter tuning with Bayesian Optimization")
