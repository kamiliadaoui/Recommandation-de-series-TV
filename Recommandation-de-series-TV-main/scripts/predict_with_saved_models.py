import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.clustering import KMeansModel

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

spark = (
    SparkSession.builder
    .appName("Prediction avec modèles sauvegardés")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")
df = df.filter(col("decade").isNotNull())

shows = df.groupBy(
    "show_id", "name", "popularity", "vote_average",
    "vote_count", "number_of_seasons",
    "number_of_episodes", "episode_run_time", "decade", "adult"
).pivot("genre_name").agg(lit(1)).na.fill(0)

shows = shows.filter(
    (col("episode_run_time") < 180) &
    (col("episode_run_time") >= 0) &
    (col("vote_count") > 0) &
    col("decade").isNotNull() &
    (col("adult") == 0)
)

shows = shows.withColumn(
    "serie_length",
    when(col("number_of_seasons") == 1, 1)
    .when(col("number_of_seasons") <= 3, 2)
    .when(col("number_of_seasons") <= 7, 3)
    .otherwise(4)
)

shows = shows.withColumn(
    "vote_quality",
    col("vote_average") * (col("vote_count") / (col("vote_count") + 100))
)

shows = shows.withColumn(
    "content_density",
    when(
        col("number_of_seasons") > 0,
        col("number_of_episodes") / col("number_of_seasons")
    ).otherwise(0)
)

excluded_cols = [
    "show_id", "name", "vote_average", "number_of_seasons",
    "number_of_episodes", "episode_run_time", "adult", "null"
]

with open("outputs/feature_cols.json", encoding="utf-8") as f:
    feature_cols = json.load(f)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

assembled = assembler.transform(shows)

scaler = StandardScalerModel.load(
    f"{HDFS_PATH}/models/scaler_model"
)

kmeans = KMeansModel.load(
    f"{HDFS_PATH}/models/kmeans_model"
)

scaled = scaler.transform(assembled)
predictions = kmeans.transform(scaled)

print("Prédictions effectuées avec les modèles rechargés.")

predictions.select(
    "show_id", "name", "prediction"
).show(20, truncate=False)

predictions.groupBy("prediction").count().orderBy("prediction").show()

spark.stop()
