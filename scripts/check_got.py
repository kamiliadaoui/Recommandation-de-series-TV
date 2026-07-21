import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("check") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"
df = spark.read.parquet(f"{HDFS_PATH}/shows_clustered")

# Cluster de GOT
print("=== CLUSTER DE GOT ===")
got = df.filter(df.name == "Game of Thrones")
got.select("name", "cluster_id").show()

cluster_id = got.collect()[0]["cluster_id"]

# Taille du cluster
print(f"=== TAILLE DU CLUSTER {cluster_id} ===")
df.filter(df.cluster_id == cluster_id).count()

# Séries similaires dans le cluster
print(f"=== SERIES DU CLUSTER {cluster_id} (top 20 par vote_quality) ===")
df.filter(df.cluster_id == cluster_id) \
  .orderBy("vote_quality", ascending=False) \
  .select("name", "vote_quality", "cluster_id") \
  .show(20, truncate=False)

spark.stop()