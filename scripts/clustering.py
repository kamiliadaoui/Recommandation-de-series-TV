import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ============================================================
# 1. DEMARRER SPARK
# ============================================================
spark = SparkSession.builder \
    .appName("TV Series - Clustering") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ============================================================
# 2. LIRE LES DONNEES NETTOYEES
# ============================================================
HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"
df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")
print(f"Nombre de lignes de départ : {df.count()}")

# ============================================================
# 3. AGREGATION PAR SHOW_ID + ONE-HOT ENCODING DES GENRES
# ============================================================
shows_genres = df.groupBy(
    "show_id", "name", "popularity", "vote_average",
    "vote_count", "number_of_seasons",
    "number_of_episodes", "episode_run_time", "decade", "adult"
).pivot("genre_name").agg(lit(1)).na.fill(0)

print(f"Nombre de séries après agrégation : {shows_genres.count()}")

# ============================================================
# 4. FILTRAGE DES OUTLIERS
# ============================================================
shows_genres = shows_genres.filter(
    (col("episode_run_time") < 180) & (col("episode_run_time") >= 0)
)
shows_genres = shows_genres.filter(col("vote_count") > 0)
shows_genres = shows_genres.filter(col("decade").isNotNull())
shows_genres = shows_genres.filter(col("adult") == 0)

print(f"Nombre de séries après filtrage : {shows_genres.count()}")

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
# 1 = mini-série, 2 = courte, 3 = moyenne, 4 = longue
shows_genres = shows_genres.withColumn(
    "serie_length",
    when(col("number_of_seasons") == 1, 1)
    .when(col("number_of_seasons") <= 3, 2)
    .when(col("number_of_seasons") <= 7, 3)
    .otherwise(4)
)

# Note pondérée par le nombre de votes (formule IMDb)
shows_genres = shows_genres.withColumn(
    "vote_quality",
    col("vote_average") * (col("vote_count") / (col("vote_count") + 100))
)

# Nombre moyen d'épisodes par saison
shows_genres = shows_genres.withColumn(
    "content_density",
    when(col("number_of_seasons") > 0,
         col("number_of_episodes") / col("number_of_seasons"))
    .otherwise(0)
)

shows_genres.cache()

# ============================================================
# 6. PREPARER LES FEATURES POUR KMEANS
# ============================================================
excluded_cols = ["show_id", "name", "vote_average", "number_of_seasons",
                 "number_of_episodes", "episode_run_time", "adult", "null"]

feature_cols = [c for c in shows_genres.columns if c not in excluded_cols]

print(f"\nFeatures utilisées : {feature_cols}")
print(f"Nombre total de features : {len(feature_cols)}")

# ============================================================
# 7. VECTOR ASSEMBLER
# ============================================================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)
data_assembled = assembler.transform(shows_genres)

# ============================================================
# 8. NORMALISATION
# ============================================================
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(data_assembled)
data_scaled = scaler_model.transform(data_assembled)
data_scaled.cache()

# ============================================================
# 9. ENTRAINER LE MODELE KMEANS
# K=12 choisi suite aux expérimentations dans experiments.py
# ============================================================
K_OPTIMAL = 12
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")

kmeans = KMeans(featuresCol="features", k=K_OPTIMAL, seed=42)
model = kmeans.fit(data_scaled)
predictions = model.transform(data_scaled)
predictions.cache()

# ============================================================
# 10. EVALUER LE MODELE
# ============================================================
silhouette = evaluator.evaluate(predictions)
print(f"\n=== SCORE DE SILHOUETTE : {silhouette:.4f} ===")
print("(plus proche de 1 = clusters bien séparés, proche de 0 = clusters qui se chevauchent)")

print("\n=== TAILLE DE CHAQUE CLUSTER ===")
predictions.groupBy("prediction").count().orderBy("prediction").show()

# ============================================================
# 11. SAUVEGARDER SUR HDFS
# ============================================================
result = predictions.select(
    "show_id", "name", "popularity", "vote_quality",
    "vote_count", "decade", "serie_length",
    "content_density", "prediction"
).withColumnRenamed("prediction", "cluster_id")

result.write.mode("overwrite").parquet(f"{HDFS_PATH}/shows_clustered")

print("\n✅ Clustering terminé et sauvegardé sur HDFS !")

spark.stop()