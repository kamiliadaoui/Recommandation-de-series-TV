import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder \
    .appName("TV Series - KMeans Clustering") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/tvshows/input"

df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")

print("=== APERCU DES DONNEES ===")
df.show(5)
print(f"Nombre de lignes : {df.count()}")

df_ml = df.select(
    "show_id", "name", "popularity", "number_of_seasons",
    "number_of_episodes", "eposide_run_time", "vote_average",
    "vote_count", "genre_name", "status_name"
).na.drop(subset=[
    "popularity", "number_of_seasons", "number_of_episodes",
    "eposide_run_time", "vote_average", "vote_count"
])

df_ml = df_ml.dropDuplicates(["show_id"])

print(f"\nNombre de séries après nettoyage : {df_ml.count()}")

indexer_genre = StringIndexer(inputCol="genre_name", outputCol="genre_index", handleInvalid="keep")
indexer_status = StringIndexer(inputCol="status_name", outputCol="status_index", handleInvalid="keep")

df_ml = indexer_genre.fit(df_ml).transform(df_ml)
df_ml = indexer_status.fit(df_ml).transform(df_ml)

feature_cols = [
    "popularity", "number_of_seasons", "number_of_episodes",
    "eposide_run_time", "vote_average", "vote_count",
    "genre_index", "status_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df_ml = assembler.transform(df_ml)

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
df_ml = scaler.fit(df_ml).transform(df_ml)

print("\n RECHERCHE DU MEILLEUR K (Méthode du coude + Silhouette) ")
print(f"{'K':>3} | {'Silhouette Score':>17} | {'Inertie (WSSSE)':>17}")
print("-" * 45)

evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
results = []

for k in range(2, 8):
    kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=20)
    model = kmeans.fit(df_ml)
    predictions = model.transform(df_ml)

    silhouette = evaluator.evaluate(predictions)
    inertia = model.summary.trainingCost

    results.append((k, silhouette, inertia))
    print(f"{k:>3} | {silhouette:>17.4f} | {inertia:>17.2f}")

best_k, best_silhouette, _ = max(results, key=lambda x: x[1])
print(f"\n>>> Meilleur K = {best_k} (Silhouette = {best_silhouette:.4f})")
print(f"\n=== MODELE FINAL AVEC K = {best_k} ===")

kmeans_final = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=20)
model_final = kmeans_final.fit(df_ml)
df_result = model_final.transform(df_ml)

silhouette_final = evaluator.evaluate(df_result)
print(f"Silhouette Score final : {silhouette_final:.4f}")

print("\n=== TAILLE DES CLUSTERS ===")
df_result.groupBy("prediction") \
    .count() \
    .orderBy("prediction") \
    .show()

print("=== PROFIL DES CLUSTERS (moyennes) ===")
df_result.groupBy("prediction").avg(
    "popularity", "number_of_seasons", "number_of_episodes",
    "eposide_run_time", "vote_average", "vote_count"
).orderBy("prediction").show()

print("=== EXEMPLES DE SERIES PAR CLUSTER ===")
for i in range(best_k):
    print(f"\n--- Cluster {i} ---")
    df_result.filter(col("prediction") == i) \
        .select("name", "popularity", "vote_average", "vote_count",
                "number_of_seasons", "genre_name") \
        .orderBy(col("popularity").desc()) \
        .show(5, truncate=False)

df_result.select(
    "show_id", "name", "popularity", "number_of_seasons",
    "number_of_episodes", "vote_average", "vote_count",
    "genre_name", "status_name", "prediction"
).write.mode("overwrite").parquet(f"{HDFS_PATH}/shows_clustered")

print("\nRésultats sauvegardés sur HDFS")

spark.stop()
