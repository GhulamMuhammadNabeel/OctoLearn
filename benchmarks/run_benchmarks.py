import time
import pandas as pd
from sklearn.datasets import fetch_california_housing
from octolearn import AutoML
import os

def run_california_benchmark():
    print("=== OctoLearn Benchmark: California Housing ===")
    
    # 1. Load Data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} features")
    
    # 2. Initialize AutoML
    # We use a limited n_trials and timeout for quick benchmarking
    automl = AutoML(
        train_models=True,
        use_optuna=True,
        n_trials=10,
        timeout_seconds=60,
        generate_report=True,
        parallel_processing=True
    )
    
    # 3. Fit Pipeline
    start_time = time.time()
    print("\nStarting pipeline fit...")
    automl.fit(X, y)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nPipeline fit completed in {duration:.2f} seconds.")
    
    # 4. Results
    best_model = automl.get_best_model()
    print(f"Best Model: {type(best_model).__name__}")
    
    leaderboard = automl.get_model_comparison()
    print("\nLeaderboard:")
    print(leaderboard)
    
    risk_score = automl.get_risk_score()
    print(f"\nDataset Risk Score: {risk_score}")
    
    print(f"\nReport generated at: {os.path.abspath('octolearn_report.pdf')}")
    print("===============================================")

if __name__ == "__main__":
    run_california_benchmark()
