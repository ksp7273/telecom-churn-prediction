# Customer Churn Prediction for Telecom

This document outlines the process for predicting customer churn using Kaggle’s Telco Customer Churn dataset, including data setup in GitHub Codespace, model training with XGBoost, deployment on Azure Machine Learning, and visualization in Power BI.

## 1. Steps to Get Kaggle’s Telco Customer Churn Dataset into GitHub Codespace

1. **Obtain Kaggle API Credentials**:
   - Log in to [Kaggle](https://www.kaggle.com).
   - Go to Profile > Account > API > Create New API Token.
   - Download `kaggle.json` (contains `username` and `key`).

2. **Set Up GitHub Codespace**:
   - Create a new repository on [GitHub](https://github.com) (e.g., `telecom-churn-prediction`).
   - Click **Code** > **Codespaces** > **Create codespace on main**.

3. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

4. **Configure Kaggle API**:
   - Upload `kaggle.json` to Codespace via file explorer.
   - Move to the correct directory:
     ```bash
     mkdir -p ~/.kaggle
     mv kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Alternatively, set environment variables:
     ```bash
     export KAGGLE_USERNAME=your_kaggle_username
     export KAGGLE_KEY=your_kaggle_api_key
     ```

5. **Download the Dataset**:
   ```bash
   kaggle datasets download -d blastchar/telco-customer-churn
   unzip telco-customer-churn.zip
   ```
   - This extracts `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

6. **Verify the Dataset**:
   ```bash
   ls
   ```
   - Optionally, check in Python:
     ```python
     import pandas as pd
     df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
     print(df.head())
     ```

7. **Optional: Push to GitHub**:
   ```bash
   git add WA_Fn-UseC_-Telco-Customer-Churn.csv
   git commit -m "Add Telco Customer Churn dataset"
   git push origin main
   ```

## 2. How `churn_prediction.py` Works

The `churn_prediction.py` script uses the Kaggle dataset and performs:
- **Loads Data**: Reads `WA_Fn-UseC_-Telco-Customer-Churn.csv`, converts `TotalCharges` to numeric, fills missing values.
- **Feature Engineering**: Creates tenure buckets, usage patterns, and encodes categorical variables.
- **Model Training**: Trains an XGBoost classifier, evaluates metrics (accuracy, precision, recall, ROC AUC).
- **SHAP Analysis**: Interprets feature importance, saves a plot.
- **Saves Outputs**: Saves the model (`churn_model.json`) and processed data (`processed_churn_data.csv`).

```python
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
```

**Run Instructions**:
- Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the Codespace directory.
- Install dependencies:
  ```bash
  pip install pandas numpy scikit-learn xgboost shap matplotlib
  ```
- Run:
  ```bash
  python churn_prediction.py
  ```
- Outputs: `churn_model.json`, `processed_churn_data.csv`, `shap_feature_importance.png`.
- Push to GitHub:
  ```bash
  git add churn_prediction.py
  git commit -m "Updated churn_prediction.py with Kaggle dataset"
  git push origin main
  ```

## 3. Model Used and Why

**Model**: XGBoost (eXtreme Gradient Boosting)

**Why XGBoost?**:
- **Performance**: Excels in tabular data tasks due to handling non-linear relationships and feature interactions.
- **Imbalanced Data**: Supports class weighting for imbalanced datasets like churn.
- **Feature Importance**: Provides built-in scores and SHAP compatibility for interpretability.
- **Scalability**: Efficient for medium-sized datasets (~7,000 rows).
- **Robustness**: Handles missing values and categorical features well.

**Alternatives**:
- Logistic Regression: Simpler but less effective for non-linear patterns.
- Random Forest: Comparable but often less accurate.
- Neural Networks: Overkill for this dataset size, less interpretable.

## 4. XGBoost Model Architecture and Workflow

**Architecture**:
- **Type**: Ensemble of gradient boosted decision trees.
- **Components**:
  - **Decision Trees**: Built sequentially, each correcting prior errors.
  - **Gradient Boosting**: Minimizes log-loss via gradient descent.
  - **Regularization**: L1/L2 penalties to prevent overfitting.
  - **Parameters**: `use_label_encoder=False`, `eval_metric='logloss'`.

**Workflow**:
1. **Input Data**: Features (tenure, MonthlyCharges, encoded categoricals) and target (Churn: Yes/No).
2. **Data Splitting**: 80% training, 20% testing (random_state=42).
3. **Training**:
   - Initialize a weak learner (decision tree).
   - Add trees iteratively to minimize log-loss.
   - Apply regularization.
4. **Prediction**: Combine tree outputs for churn probability (threshold 0.5).
5. **Evaluation**: Compute accuracy, precision, recall, ROC AUC.
6. **SHAP Analysis**: Quantify feature contributions.

**Feature Engineering**:
- **Tenure Buckets**: Groups tenure (e.g., 0-1yr, 1-2yr).
- **Usage Patterns**: `MonthlyToTotalChargeRatio`, `AvgMonthlyCharge`.
- **Categorical Encoding**: Label encoding for categorical variables.

## 5. Pros and Cons of XGBoost

**Pros**:
- High accuracy for tabular data.
- Feature importance insights.
- Flexible and scalable.
- SHAP integration for interpretability.

**Cons**:
- Complex to tune.
- Overfitting risk without tuning.
- Slower than simpler models.
- Less intuitive than linear models.

## 6. Deploy on Azure Machine Learning

**Prerequisites**:
- Azure subscription.
- Azure ML workspace (`churn-workspace`).
- `churn_model.json`.
- Install Azure ML SDK:
  ```bash
  pip install azure-ai-ml azure-identity
  az login
  ```

**Steps**:
1. **Set Up Workspace**:
   - In [Azure Portal](https://portal.azure.com), create a Machine Learning workspace.
   - Download `config.json` from workspace > Overview.
   - Upload `config.json` to Codespace.

2. **Create Deployment Script**:
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
from azure.identity import DefaultAzureCredential
import os

# Authenticate and connect to Azure ML workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Register the model
model = Model(
    path="churn_model.json",
    name="churn_model",
    type="custom_model",
    description="XGBoost model for telecom churn prediction"
)
registered_model = ml_client.models.create_or_update(model)

# Create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="churn-endpoint",
    description="Endpoint for telecom churn prediction",
    auth_mode="key"
)
ml_client.online_endpoints.create_or_update(endpoint)

# Create scoring script
scoring_script = """
import json
import xgboost as xgb
import pandas as pd
import numpy as np

def init():
    global model
    model = xgb.Booster()
    model.load_model('churn_model.json')

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        dmatrix = xgb.DMatrix(df)
        predictions = model.predict(dmatrix)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
"""

with open("score.py", "w") as f:
    f.write(scoring_script)

# Create environment
environment = Environment(
    name="churn-env",
    conda_file={
        "dependencies": [
            {"pip": ["xgboost==1.7.3", "pandas", "numpy"]}
        ]
    },
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)
ml_client.environments.create_or_update(environment)

# Deploy the model
deployment = ManagedOnlineDeployment(
    name="churn-deployment",
    endpoint_name="churn-endpoint",
    model=registered_model,
    environment=environment,
    code_configuration={
        "code": ".",  # Current directory containing score.py
        "scoring_script": "score.py"
    },
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.create_or_update(deployment)

# Get endpoint scoring URI
endpoint = ml_client.online_endpoints.get(name="churn-endpoint")
print(f"Scoring URI: {endpoint.scoring_uri}")
print(f"Primary key: {ml_client.online_endpoints.get_keys(name='churn-endpoint').primary_key}")
```

3. **Run Deployment**:
   ```bash
   python deploy_azure.py
   ```

4. **Test Endpoint**:
```python
import requests
import json
import pandas as pd

# Sample data
sample_data = pd.read_csv('processed_churn_data.csv').drop(['customerID', 'Churn'], axis=1).iloc[0:1]
sample_json = sample_data.to_dict(orient='records')

# Endpoint details
scoring_uri = "YOUR_SCORING_URI"
key = "YOUR_PRIMARY_KEY"

# Send request
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
response = requests.post(scoring_uri, json=sample_json, headers=headers)
print(response.json())
```

5. **Push to GitHub**:
   ```bash
   git add deploy_azure.py test_endpoint.py
   git commit -m "Add Azure deployment and test scripts"
   git push origin main
   ```

## 7. Create Power BI Dashboards

1. **Import Data**:
   - Open Power BI Desktop.
   - Get Data > Text/CSV > Select `processed_churn_data.csv`.

2. **Create Visualizations**:
   - **Churn Rate by Tenure Bucket**: Bar Chart (X: `TenureBucket`, Y: Count of `Churn`).
   - **Feature Importance**: Image visual with `shap_feature_importance.png`.
   - **Churn by Contract Type**: Pie Chart (Legend: `Contract`, Values: Count of `Churn`).
   - **Monthly Charges vs. Churn**: Scatter Plot (X: `MonthlyCharges`, Y: `TotalCharges`, Color: `Churn`).
   - Add slicers for `InternetService`, `PaymentMethod`.

3. **DAX Measures**:
   ```dax
   Total Churn = CALCULATE(COUNTROWS(Table), Table[Churn] = 1)
   Churn Percentage = DIVIDE([Total Churn], COUNTROWS(Table)) * 100
   ```

4. **Publish**:
   - Save as `churn_dashboard.pbix`.
   - Publish to Power BI Service.
   - Share the dashboard URL.

5. **Push to GitHub** (Optional):
   ```bash
   git add churn_dashboard.pbix
   git commit -m "Add Power BI dashboard file"
   git push origin main
   ```

## Summary
- **Dataset**: Integrated via Kaggle API.
- **Model**: XGBoost for performance and interpretability.
- **Deployment**: Azure ML endpoint for predictions.
- **Power BI**: Visualizes churn insights.
- **GitHub**: All code and outputs versioned.