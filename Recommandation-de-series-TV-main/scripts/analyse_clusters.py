from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, min, max, round as spark_round
from pathlib import Path

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

spark = (
    SparkSession.builder
    .appName("Analyse des clusters")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(f"{HDFS_PATH}/shows_clustered")

summary = (
    df.groupBy("cluster_id")
    .agg(
        count("*").alias("nb_series"),
        spark_round(avg("popularity"), 2).alias("popularite_moyenne"),
        spark_round(avg("vote_quality"), 2).alias("qualite_vote_moyenne"),
        spark_round(avg("vote_count"), 2).alias("votes_moyens"),
        min("decade").alias("decennie_min"),
        max("decade").alias("decennie_max"),
        spark_round(avg("serie_length"), 2).alias("longueur_moyenne"),
        spark_round(avg("content_density"), 2).alias("episodes_par_saison")
    )
    .orderBy("cluster_id")
)

summary.show(20, truncate=False)

Path("outputs").mkdir(exist_ok=True)
summary.toPandas().to_csv("outputs/cluster_summary.csv", index=False)

print("Résumé sauvegardé dans outputs/cluster_summary.csv")

spark.stop()
