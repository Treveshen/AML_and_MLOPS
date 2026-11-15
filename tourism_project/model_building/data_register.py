import sys
import pandas as pd
from pathlib import Path

def register_and_validate_data(file_path: str):
    print("==================================================")
    print("STEP 1: DATA REGISTRATION & VALIDATION")
    print("==================================================")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Error: Dataset not found at path: {file_path}")
        
    df = pd.read_csv(path)
    print(f"Successfully loaded dataset from: {file_path}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data Schema Validation Failed. Missing columns: {missing_cols}")
    
    print("\n[Schema Check]: All expected columns are present.")
    print(f"[Target Balance]:\n{df['ProdTaken'].value_counts(normalize=True).rename('Percentage') * 100}")
    print("\n[Missing Values Summary]:")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0] if missing_summary.sum() > 0 else "No missing values found.")
    print("==================================================")
    print("Dataset registered and validated successfully.")
    print("==================================================")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "tourism_project/data/tourism.csv"
    register_and_validate_data(csv_file)
