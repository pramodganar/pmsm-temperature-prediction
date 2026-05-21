from sklearn.model_selection import train_test_split


class DataPreprocessing:

    def split_data(self, df):

        X = df.drop("pm", axis=1)

        y = df["pm"]

        # Fill Missing Values
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        return X_train, X_test, y_train, y_test