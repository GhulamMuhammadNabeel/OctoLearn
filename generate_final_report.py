
import pandas as pd
import seaborn as sns
import logging
from octolearn.core import AutoML, DataConfig, ProfilingConfig, ModelingConfig, ParallelConfig, OptimizationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load data
df = sns.load_dataset('titanic')

# Config
# Sample size 500
config = DataConfig(sample_size=500, use_full_data=False) 
parallel = ParallelConfig(enable_gpu=False) 
modeling = ModelingConfig(train_models=True, n_models=2) 
optimization = OptimizationConfig(use_optuna=False) # Fast

# Initialize AutoML
automl = AutoML(
    data_config=config,
    modeling_config=modeling,
    optimization_config=optimization,
    parallel_config=parallel,
    profiling_config=ProfilingConfig(detect_outliers=True, analyze_interactions=False),
    # Use defaults for title etc. or rely on default
)

# Run Fit
print("Running AutoML fit...")
X = df.drop(columns=['survived'])
y = df['survived']
automl.fit(X, y)

# Generate Report via public API
print("Generating final report: final_magazine_report.pdf")
pdf_path = automl.generate_report(filename="final_magazine_report.pdf")
print(f"Done. Saved to: {pdf_path}")
