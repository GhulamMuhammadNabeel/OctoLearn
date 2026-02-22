from octolearn import AutoML

if __name__ == "__main__":
    print("Testing surprise_me API for classification...")
    pdf_path, best_model = AutoML.surprise_me(task='classification')
    print(f"Classification Best Model: {best_model.__class__.__name__}")
    print(f"Classification PDF Path: {pdf_path}")
    
    print("\nTesting surprise_me API for regression...")
    pdf_path, best_model = AutoML.surprise_me(task='regression')
    print(f"Regression Best Model: {best_model.__class__.__name__}")
    print(f"Regression PDF Path: {pdf_path}")
