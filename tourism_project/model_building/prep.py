import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_csv: str):
    print("==================================================")
    print("STEP 2: DATA PREPARATION & SPLITTING")
    print("==================================================")

    df = pd.read_csv(input_csv)
    print(f"Loaded raw data: {df.shape}")

    # 1. Clean column discrepancies
    # In 'Gender', 'Fe Male' is cleaned to 'Female'
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

    # In 'MaritalStatus', align 'Unmarried' to 'Single'
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

    # 2. Drop unnecessary metadata identifier columns
    drop_cols = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped non-predictive columns: {drop_cols}")

    # 3. Separate features (X) and target (y)
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

    # 5. Save splits locally in current directory for artifact packaging
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("\nSaved split files: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("==================================================")
    print("Data preparation completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "tourism_project/data/tourism.csv"
    prepare_data(csv_file)
