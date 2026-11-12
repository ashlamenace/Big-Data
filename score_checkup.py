"""Déploiement (partie 2) : score en batch un fichier de bilans avec le PipelineModel Spark sauvegardé.
Usage : spark-submit score_checkup.py checkup_clean.parquet scores.parquet   (ou un chemin hdfs://...)"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder.appName("score_checkup").getOrCreate()
model = PipelineModel.load("models/checkup_spark_model")
out = (model.transform(spark.read.parquet(sys.argv[1]))
       .select("MedID", F.round(vector_to_array("probability")[1], 4).alias("proba_prediabete"), "prediction"))
out.write.mode("overwrite").parquet(sys.argv[2])
out.show(10)
