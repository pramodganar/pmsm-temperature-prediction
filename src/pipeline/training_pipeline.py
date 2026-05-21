
from src.components.data_ingestion import DataIngestion

def run_training_pipeline():
    ingestion = DataIngestion("data/raw/electric_motor_temperature.csv")
    df = ingestion.load_data()

    print(df.head())

if __name__ == "__main__":
    run_training_pipeline()
