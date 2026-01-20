from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from utils.logger import model_logger

class GeneticOptimizer:
    """
    Optimizes model hyperparameters using Genetic Algorithms.
    """

    def __init__(self, model_type: str = "random_forest", cv: int = 3, n_jobs: int = -1):
        self.model_type = model_type
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_estimator_ = None
        self.best_params_ = None
        self.history_ = None

    def _get_model_and_space(self) -> Tuple[Any, Dict]:
        """Returns the base model and the hyperparameter search space."""
        if self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5),
                'criterion': Categorical(['gini', 'entropy'])
            }
        elif self.model_type == "logistic_regression":
            model = LogisticRegression(random_state=42, solver='liblinear')
            param_grid = {
                'C': Continuous(0.01, 10.0, distribution='log-uniform'),
                'penalty': Categorical(['l1', 'l2'])
            }
        else:
            raise ValueError(f"Model {self.model_type} not supported for optimization.")
        
        return model, param_grid

    def optimize(self, X_train, y_train, population_size=20, generations=10) -> Dict[str, Any]:
        """
        Runs the genetic algorithm optimization.
        """
        model, param_grid = self._get_model_and_space()
        
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        model_logger.info(f"Starting Genetic Optimization for {self.model_type}...")
        model_logger.info(f"Population: {population_size}, Generations: {generations}")

        evolved_estimator = GASearchCV(
            estimator=model,
            cv=cv,
            scoring='accuracy',
            population_size=population_size,
            generations=generations,
            tournament_size=3,
            elitism=True,
            crossover_probability=0.8,
            mutation_probability=0.1,
            param_grid=param_grid,
            criteria='max',
            algorithm='eaMuPlusLambda',
            n_jobs=self.n_jobs,
            verbose=True,
            keep_top_k=4
        )

        evolved_estimator.fit(X_train, y_train)

        self.best_estimator_ = evolved_estimator.best_estimator_
        self.best_params_ = evolved_estimator.best_params_
        self.history_ = evolved_estimator.history
        
        model_logger.info(f"Optimization Complete. Best Score: {evolved_estimator.best_score_:.4f}")
        model_logger.info(f"Best Params: {self.best_params_}")

        return self.best_params_

    def plot_history(self, save_path: str = "models/optimization_history.png"):
        """Plots the fitness evolution over generations."""
        if self.history_ is None:
            model_logger.warning("No history to plot. Run optimize() first.")
            return

        plt.figure(figsize=(10, 6))
        # GASearchCV history is often a dict or list of dicts. 
        # sklearn-genetic-opt history format: {'fitness': [gens], 'fitness_std': [gens], ...}
        # Let's assume standard format from the lib.
        
        # Checking compatibility: older versions might use different structure. 
        # But generally it stores 'fitness' (max) and 'fitness_mean'.
        
        try:
            fitness = self.history_['fitness']
            generations = range(len(fitness))
            plt.plot(generations, fitness, label='Best Fitness')
            
            if 'fitness_mean' in self.history_:
                plt.plot(generations, self.history_['fitness_mean'], label='Mean Fitness')
                
            plt.title('Genetic Algorithm Optimization History')
            plt.xlabel('Generations')
            plt.ylabel('Fitness (Accuracy)')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()
            model_logger.info(f"Optimization history plot saved to {save_path}")
        except Exception as e:
            model_logger.error(f"Could not plot history: {e}")
