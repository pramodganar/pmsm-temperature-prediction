import joblib
import pandas as pd
import os


class PredictionPipeline:

    def __init__(self):

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_root = os.path.abspath(
            os.path.join(
                current_dir,
                "..",
                ".."
            )
        )

        model_path = os.path.join(
            project_root,
            "models",
            "best_model.pkl"
        )

        self.model = joblib.load(
            model_path
        )

        self.feature_columns = [

            "profile_id",
            "u_q",
            "coolant",
            "u_d",
            "motor_speed",
            "i_d",
            "i_q",
            "ambient",
            "current_magnitude",
            "voltage_magnitude",
            "temp_difference"
        ]

    def predict(self, data):

        data = data[self.feature_columns]

        prediction = self.model.predict(data)

        return prediction