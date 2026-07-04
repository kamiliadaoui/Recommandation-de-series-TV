import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import csv
from datetime import datetime

spark = SparkSession.builder \
    .appName("TV Series - Experiments") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"
df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")

# Même pipeline que clustering.py
shows_genres = df.groupBy(
    "show_id", "name", "popularity", "vote_average",
    "vote_count", "number_of_seasons",
    "number_of_episodes", "episode_run_time", "decade", "adult"
).pivot("genre_name").agg(lit(1)).na.fill(0)

shows_genres = shows_genres.filter(
    (col("episode_run_time") < 180) & (col("episode_run_time") >= 0)
).filter(col("vote_count") > 0) \
 .filter(col("decade").isNotNull()) \
 .filter(col("adult") == 0)

shows_genres = shows_genres.withColumn(
    "serie_length",
    when(col("number_of_seasons") == 1, 1)
    .when(col("number_of_seasons") <= 3, 2)
    .when(col("number_of_seasons") <= 7, 3)
    .otherwise(4)
).withColumn(
    "vote_quality",
    col("vote_average") * (col("vote_count") / (col("vote_count") + 100))
).withColumn(
    "content_density",
    when(col("number_of_seasons") > 0,
         col("number_of_episodes") / col("number_of_seasons"))
    .otherwise(0)
)

shows_genres.cache()

excluded_cols = ["show_id", "name", "vote_average", "number_of_seasons",
                 "number_of_episodes", "episode_run_time", "adult", "null"]
feature_cols = [c for c in shows_genres.columns if c not in excluded_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)
data_assembled = assembler.transform(shows_genres)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(data_assembled)
data_scaled = scaler_model.transform(data_assembled)
data_scaled.cache()

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction"
)

# ============================================================
# EXPERIMENTATIONS — on teste plusieurs K et seeds
# ============================================================
experiments = []

K_range = [3, 5, 8, 10, 12, 15]
seeds = [42, 123, 456]

print("\n=== DEBUT DES EXPERIMENTATIONS ===")
print(f"{'K':<5} {'Seed':<8} {'Coût':<15} {'Silhouette':<12} {'Nb séries/cluster (min/max)'}")
print("-" * 70)

for k in K_range:
    for seed in seeds:
        # Entraîner le modèle
        kmeans = KMeans(featuresCol="features", k=k, seed=seed)
        model = kmeans.fit(data_scaled)
        predictions = model.transform(data_scaled)

        # Calculer les métriques
        cost = model.summary.trainingCost
        silhouette = evaluator.evaluate(predictions)

        # Taille min et max des clusters
        cluster_sizes = predictions.groupBy("prediction").count()
        min_size = cluster_sizes.agg({"count": "min"}).collect()[0][0]
        max_size = cluster_sizes.agg({"count": "max"}).collect()[0][0]

        print(f"K={k:<3} seed={seed:<5} coût={cost:<15.2f} silhouette={silhouette:<12.4f} min={min_size} max={max_size}")

        # Sauvegarder les résultats
        experiments.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "k": k,
            "seed": seed,
            "cout": round(cost, 2),
            "silhouette": round(silhouette, 4),
            "nb_series": shows_genres.count(),
            "nb_features": len(feature_cols),
            "min_cluster_size": min_size,
            "max_cluster_size": max_size
        })

# ============================================================
# SAUVEGARDER LES RESULTATS DANS UN CSV
# ============================================================
output_file = r"C:\projet-hadoop\experiments\results.csv"
os.makedirs(r"C:\projet-hadoop\experiments", exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=experiments[0].keys())
    writer.writeheader()
    writer.writerows(experiments)

print(f"\n✅ Résultats sauvegardés dans {output_file}")

# ============================================================
# RESUME — meilleur modèle
# ============================================================
best = max(experiments, key=lambda x: x["silhouette"])
print(f"\n=== MEILLEUR MODELE ===")
print(f"K={best['k']}, seed={best['seed']}")
print(f"Silhouette={best['silhouette']}, Coût={best['cout']}")

spark.stop()