from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import StandardScalerModel

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

spark = (
    SparkSession.builder
    .appName("TestModelsLoad")
    .getOrCreate()
)

scaler_model = StandardScalerModel.load(
    f"{HDFS_PATH}/models/scaler_model"
)

kmeans_model = KMeansModel.load(
    f"{HDFS_PATH}/models/kmeans_model"
)

print("Scaler chargé avec succès.")
print("K-Means chargé avec succès.")
print("Nombre de clusters :", kmeans_model.getK())

spark.stop()