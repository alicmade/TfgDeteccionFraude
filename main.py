import os

from kagglehub import kagglehub

from pyspark.sql import SparkSession

from fraudDetectionPipeline import FraudDetectionPipeline
import shutil

if __name__ == "__main__":

    path = kagglehub.dataset_download("sriharshaeedala/financial-fraud-detection-dataset")
    spark = SparkSession.builder.appName("FraudDetection").config("spark.executor.memory", "2g").config("spark.driver.memory", "2g").config("spark.sql.shuffle.partitions", "4").getOrCreate()

    print(f"Ruta local del dataset descargado: {path}")
    print(os.listdir(path))

    pipeline = FraudDetectionPipeline(spark, path)
    pipeline.load_data()
    pipeline.preprocess()
    pipeline.train_and_evaluate_models()
    pipeline.export_results(
        csv_path="output/resultados_metricas.csv",
        sheet_url="https://docs.google.com/spreadsheets/d/1GWfUrWNkHEDb9XCGwjisNzfnoXPT8WXrOCdv4cvAC4w/edit?usp=sharing",
        creds_path="./credenciales.json"
    )

    try:
        shutil.rmtree(path)
        print(f"Datos de Kaggle eliminados correctamente de: {path}")
    except Exception as e:
        print(f"Error al eliminar los datos de Kaggle: {e}")


