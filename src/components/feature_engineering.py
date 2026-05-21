import numpy as np


class FeatureEngineering:

    def create_features(self, df):

        df["current_magnitude"] = np.sqrt(
            df["i_d"]**2 + df["i_q"]**2
        )

        df["voltage_magnitude"] = np.sqrt(
            df["u_d"]**2 + df["u_q"]**2
        )

        df["temp_difference"] = (
            df["coolant"] - df["ambient"]
        )

        return df