from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluation:

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(X_test)

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

        return {

            "MAE": mae,

            "RMSE": rmse,

            "R2 Score": r2
        }