#!/usr/bin/env python3
"""
Script to run Genetic Algorithm Optimization.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_breast_cancer_data
from data.preprocessor import DataPreprocessor
from models.optimization import GeneticOptimizer
from utils.logger import model_logger
import json

def main():
    # 1. Load Data
    X, y = load_breast_cancer_data()
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # We combine Train + Val for optimization cross-validation to have more data
    # (GASearchCV does its own CV)
    import pandas as pd
    X_opt = pd.concat([X_train, X_val])
    y_opt = pd.concat([y_train, y_val])
    
    X_opt_scaled = preprocessor.fit_transform(X_opt)
    
    # 3. Optimizers
    # We will optimize Random Forest as it has more interesting hyperparameters for GA
    optimizer = GeneticOptimizer(model_type="random_forest", cv=3)
    
    # Run optimization (small generation count for demo speed/testing)
    best_params = optimizer.optimize(X_opt_scaled, y_opt.values, population_size=10, generations=5)
    
    # 4. Save Results
    optimizer.plot_history("models/optimization_history.png")
    
    results = {
        "best_params": best_params,
        "best_score": optimizer.best_estimator_.score(X_opt_scaled, y_opt.values) # roughly
    }
    
    with open("models/optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    model_logger.info("Optimization results saved.")

if __name__ == "__main__":
    main()
