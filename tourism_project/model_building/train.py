import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

def train_and_tune_model():
    print("==================================================")
    print("STEP 3: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("==================================================")

    # 1. Load train and test splits from artifacts
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    print(f"Dataset successfully loaded. X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 2. Identify numerical and categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # 3. Define Preprocessing Pipeline
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        (StandardScaler(), numerical_cols),
        remainder='passthrough'
    )

    # Calculate class imbalance weighting factor
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    # 4. Construct End-to-End Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos
        ))
    ])

    # 5. Define Parameter Grid for XGBoost Tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0]
    }

    # 6. Set MLflow Tracking URI and Experiment
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("Wellness_Tourism_Purchase_Prediction")

    with mlflow.start_run(run_name="XGBoost_GridSearchCV"):
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        print("Starting GridSearchCV tuning...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"\nOptimal Hyperparameters: {best_params}")

        # Log all tuned parameters to MLflow
        for param, val in best_params.items():
            mlflow.log_param(param, val)

        # Evaluate Best Model on Test Set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_rec = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)

        # Log Metrics
        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_metric("precision", test_prec)
        mlflow.log_metric("recall", test_rec)
        mlflow.log_metric("f1_score", test_f1)
        mlflow.log_metric("roc_auc", test_auc)

        print("\n=== Test Performance Report ===")
        print(f"Accuracy : {test_acc:.4f}")
        print(f"Precision: {test_prec:.4f}")
        print(f"Recall   : {test_rec:.4f}")
        print(f"F1-Score : {test_f1:.4f}")
        print(f"ROC-AUC  : {test_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Log Model artifact with MLflow
        mlflow.sklearn.log_model(best_model, "best_xgboost_model")

        # 7. Persist Best Model to deployment directory for Git Commit
        deploy_dir = "tourism_project/deployment"
        os.makedirs(deploy_dir, exist_ok=True)
        model_path = os.path.join(deploy_dir, "model.joblib")
        joblib.dump(best_model, model_path)
        print(f"\nModel pipeline committed to: {model_path}")
        print("==================================================")

if __name__ == "__main__":
    train_and_tune_model()
