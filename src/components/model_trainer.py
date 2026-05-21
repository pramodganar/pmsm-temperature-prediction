from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd


class ModelTrainer:

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            y_pred
        )

        return mae, rmse, r2

    def train_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        # Dictionary of Models
        models = {

            "Linear Regression":
            LinearRegression(),

            "Ridge Regression":
            Ridge(),

            "Random Forest":
            RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),

            "XGBoost":
            XGBRegressor(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1
            ),

            "LightGBM":
            LGBMRegressor(
                random_state=42,
                n_estimators=100
            ),

            "CatBoost":
            CatBoostRegressor(
                verbose=0,
                random_state=42,
                iterations=100
            )
        }

        results = {}

        trained_models = {}

        # Train Every Model
        for name, model in models.items():

            print(f"\nTraining {name}...")

            # Train Model
            model.fit(
                X_train,
                y_train
            )

            # Evaluate Model
            mae, rmse, r2 = self.evaluate_model(
                model,
                X_test,
                y_test
            )

            # Store Results
            results[name] = {

                "MAE": round(mae, 4),

                "RMSE": round(rmse, 4),

                "R2 Score": round(r2, 4)
            }

            # Store Trained Model
            trained_models[name] = model

            print(f"{name} Completed")

            print(f"MAE: {mae:.4f}")

            print(f"RMSE: {rmse:.4f}")

            print(f"R2 Score: {r2:.4f}")

        # Convert Results to DataFrame
        results_df = pd.DataFrame(results).T

        # Sort by Best R2 Score
        results_df = results_df.sort_values(
            by="R2 Score",
            ascending=False
        )

        return results_df, trained_models