import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os

# 1. Load and preprocess the Telco Customer Churn dataset
def load_data():
    # Load Kaggle's Telco Customer Churn dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    return df

# 2. Feature Engineering
def feature_engineering(df):
    # Create tenure buckets
    df['TenureBucket'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 36, 48, 60, np.inf], 
                               labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr'])
    
    # Create usage patterns
    df['MonthlyToTotalChargeRatio'] = df['MonthlyCharges'] / df['TotalCharges']
    df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != 'customerID':
            df[col] = le.fit_transform(df[col])
    
    return df

# 3. Train XGBoost model
def train_model(df):
    # Prepare features and target
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    
    return model, X_train, X_test, y_test

# 4. SHAP Analysis for feature importance
def shap_analysis(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.savefig('shap_feature_importance.png')
    plt.close()

# 5. Save model
def save_model(model):
    model.save_model('churn_model.json')

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Train and evaluate model
    model, X_train, X_test, y_test = train_model(df)
    
    # Perform SHAP analysis
    shap_analysis(model, X_train)
    
    # Save model
    save_model(model)
    
    # Save processed dataset for Power BI
    df.to_csv('processed_churn_data.csv', index=False)