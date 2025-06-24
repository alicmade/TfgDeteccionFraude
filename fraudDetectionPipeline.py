import time
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
    GBTClassifier,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from resultadosFinalesExportacion import resultadosFinalesExportacion


class FraudDetectionPipeline:
    def __init__(self, spark, path):
        self.spark = spark
        self.path = path
        self.df = None
        #self.model = None
        self.models = {}
        self.metrics = []

    def load_data(self):

        csv_file = [f for f in os.listdir(self.path) if f.endswith(".csv")][0]
        self.df = self.spark.read.csv(os.path.join(self.path, csv_file), header=True, inferSchema=True)
        print("ya ha cargado el dataset")
        self.df.show()
        self.df.printSchema()
        self.df.describe().show()
        self.df.groupBy("isFraud").count().show()

    def preprocess(self):

        fraud = self.df.filter("isFraud = 1")
        non_fraud = self.df.filter("isFraud = 0").sample(fraction=0.0015, seed=42)

        #mostramos el número de filas antes del balanceo y después
        balanced_df = fraud.union(non_fraud)
        self.df = balanced_df
        self.df.groupBy("isFraud").count().show()
        # Añade la columna objetivo 'label'
        self.df = self.df.withColumn("label", col("isFraud").cast("integer"))

        # Codifica la columna 'type' (es string pero relevante)
        indexer = StringIndexer(inputCol="type", outputCol="type_indexed")
        self.df = indexer.fit(self.df).transform(self.df)

        # Excluye manualmente las columnas string no útiles
        exclude_cols = ["isFraud", "label", "type", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest" ]

        # Toma solo columnas numéricas más la columna codificada
        cols = [c.name for c in self.df.schema.fields
                if str(c.dataType) != "StringType" and c.name not in exclude_cols]
        print("Columnas usadas como features:", cols)
        # Vectoriza
        assembler = VectorAssembler(inputCols=cols, outputCol="features")
        self.df = assembler.transform(self.df).select("features", "label")
        #from pyspark import StorageLevel

       #self.df.persist(StorageLevel.DISK_ONLY)

        print("ya se ha preprocesado el dataset")

#probar con mas modelos
    def train_and_evaluate_models(self):

        train, test = self.df.randomSplit([0.7, 0.3], seed=42)
        #evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
        aucpr_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
        aucroc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="precisionByLabel")
        recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        classifiers = {
            "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100),
            "LogisticRegression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=20),
            "GBTClassifier": GBTClassifier(labelCol="label", featuresCol="features", maxIter=50),
            "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features")
        }


        for name, clf in classifiers.items():
            start_time = time.time()
            model = clf.fit(train)
            predictions = model.transform(test)
            #auc = evaluator.evaluate(predictions)
            auc_pr = aucpr_evaluator.evaluate(predictions)
            auc_roc = aucroc_evaluator.evaluate(predictions)
            precision = precision_evaluator.evaluate(predictions)
            recall = recall_evaluator.evaluate(predictions)
            f1 = f1_evaluator.evaluate(predictions)

            elapsed = time.time() - start_time
            self.models[name] = model
            self.metrics.append({
                 "model": name,
                 "AUC_PR": round(auc_pr, 4),
                 "AUC_ROC": round(auc_roc, 4),
                 "Precision": round(precision, 4),
                 "Recall": round(recall, 4),
                 "F1": round(f1, 4),
                 "Time(s)": round(elapsed, 2)
            })
            #print(f"{name} AUC: {auc:.4f} | Tiempo: {elapsed:.2f} s")
            print(f"\n{name} Results:")
            print(f"AUC-PR  : {auc_pr:.4f}")
            print(f"AUC-ROC : {auc_roc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")
            print(f"Training Time: {elapsed:.2f} s")


    def export_results(self, csv_path="resultados_metricas.csv", sheet_url=None, creds_path="./credenciales.json"):
          # Crear una instancia
        exportador = resultadosFinalesExportacion(self.metrics)

        # Llamar a un método
        exportador.save_metrics_to_csv(csv_path)
        exportador.upload_csv_to_google_sheets(csv_path, sheet_url, creds_path)



