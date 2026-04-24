import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import time

def load_data(filepath):
    """
    Loads the PaySim dataset from a CSV file.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        
        print("\n--- Data Exploration ---")
        print(f"1. Dataset Shape (rows, columns): {df.shape}")
        
        print("\n2. Column Names and Data Types:")
        print(df.dtypes)
        
        print("\n3. Missing Values:")
        print(df.isnull().sum())
        
        print("\n4. Count of Fraud vs Non-Fraud Transactions:")
        print(df['isFraud'].value_counts())
        
        print("\n5. Transaction Types Containing Fraud:")
        fraud_transactions = df[df['isFraud'] == 1]
        print(fraud_transactions['type'].value_counts())
        
        print("\n6. Plotting Fraud vs Non-Fraud Distribution...")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='isFraud')
        plt.title('Fraud vs Non-Fraud Distribution')
        plt.xlabel('Is Fraud (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.savefig('fraud_distribution.png')
        print("Plot saved as 'fraud_distribution.png'")
        plt.show(block=False) 
        
        return df
        
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        print("Please ensure the PaySim CSV dataset is placed in the correct directory.")
        return None

def preprocess_data(df):
    """
    Preprocesses the dataset.
    Filters transactions, encodes categorical variables, creates new features, 
    splits the data, and applies SMOTE.
    """
    print("\nPreprocessing data...")
    
    # 1. Filter only TRANSFER and CASH_OUT transactions
    print("Filtering for TRANSFER and CASH_OUT transactions...")
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    # 2. Encode the type column (TRANSFER=0, CASH_OUT=1)
    df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})
    
    # 3. Create two new features
    print("Creating new features: errorBalanceOrig and errorBalanceDest...")
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 4. Select only these features
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest']
    X = df[features]
    y = df['isFraud']
    
    # 5. Split the data into 80% train and 20% test sets
    print("Splitting data into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Apply SMOTE to balance the training data
    print("Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_model(X_train, y_train):
    """
    Trains an XGBoost classifier for fraud detection.
    """
    print("\nTraining XGBoost model...")
    start_time = time.time()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training complete! (Time taken: {training_time:.2f} seconds)")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model with Classification Report, AUC-ROC, Confusion Matrix, 
    and Feature Importance. Saves the model to disk.
    """
    print("\n--- Model Evaluation ---")
    
    # 1. Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.4f}")
    
    # 4. Plot Confusion Matrix Heatmap
    print("\nSaving Confusion Matrix as 'confusion_matrix.png'...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # 5. Feature Importance
    print("Saving Feature Importance chart as 'feature_importance.png'...")
    feature_importances = model.feature_importances_
    features = X_test.columns
    
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png', bbox_inches='tight')
    plt.close()
    
    # 6. Print Top 3 Features
    print("\nTop 3 Most Important Features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"- {row['Feature']}: {row['Importance']:.4f}")
        
    # 7. Save Model
    model_save_path = r'D:\dell\fraud_detection\fraud_model.pkl'
    print(f"\nSaving trained model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    print("Model saved successfully!")

def main():
    # Define the path to your PaySim CSV file
    dataset_path = r'D:\dell\fraud_detection\PS_20174392719_1491204439457_log.csv' 
    
    # 1. Load Data
    df = load_data(dataset_path)
    
    if df is None:
        return # Exit if data failed to load
    
    # Uncomment the sections below as you build out the pipeline
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("\n--- Data Shapes after Preprocessing ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 3. Train Model
    model = train_model(X_train, y_train)
    
    # 4. Evaluate Model
    evaluate_model(model, X_test, y_test)
    
    print("Fraud detection project pipeline execution finished.")

if __name__ == "__main__":
    main()
