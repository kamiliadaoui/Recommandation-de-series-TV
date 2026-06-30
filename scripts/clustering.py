import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# 1. DEMARRER SPARK

spark = SparkSession.builder \
    .appName("TV Series - Clustering") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. LIRE LES DONNEES NETTOYEES (parquet sauvegardé avant)

HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"

df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")

print(f"Nombre de lignes de départ : {df.count()}")

# 3. AGREGATION PAR SHOW_ID + ONE-HOT ENCODING DES GENRES

shows_genres = df.groupBy(
    "show_id", "name", "popularity", "vote_average",
    "vote_count", "number_of_seasons",
    "number_of_episodes", "episode_run_time"
).pivot("genre_name").agg(lit(1)).na.fill(0)

print(f"Nombre de séries après agrégation : {shows_genres.count()}")
print("\n=== APERCU APRES ONE-HOT ENCODING ===")
shows_genres.show(5)

print("\n=== COLONNES DISPONIBLES ===")
print(shows_genres.columns)

# 4. NETTOYAGE DES VALEURS ABERRANTES 

shows_genres = shows_genres.filter(
    (col("episode_run_time") < 180) & (col("episode_run_time") >= 0)
)

shows_genres = shows_genres.filter(col("vote_count") > 0)

print(f"\nNombre de séries après filtrage des outliers : {shows_genres.count()}")

# On garde le DataFrame en mémoire pour éviter de tout recalculer à chaque étape
shows_genres.cache()


# 5. PREPARER LES FEATURES POUR KMEANS (réduites)
numeric_cols = [
    "popularity", "vote_average", "vote_count"
]

excluded_cols = ["show_id", "name", "number_of_seasons",
                  "number_of_episodes", "episode_run_time"] + numeric_cols
genre_cols = [c for c in shows_genres.columns if c not in excluded_cols and c != "null"]

print(f"\nColonnes de genres détectées : {genre_cols}")

feature_cols = numeric_cols + genre_cols


# 6. VECTOR ASSEMBLER - combine toutes les features en un vecteur

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

data_assembled = assembler.transform(shows_genres)


# 7. NORMALISATION avec StandardScaler

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(data_assembled)
data_scaled = scaler_model.transform(data_assembled)

# On cache aussi ce DataFrame car il est réutilisé pour tester plusieurs K
data_scaled.cache()

print("\n=== DONNEES PRETES POUR KMEANS ===")
data_scaled.select("show_id", "name", "features").show(5, truncate=False)


# 8. METHODE DU COUDE - trouver le meilleur K

print("\n=== METHODE DU COUDE (test de plusieurs K) ===")
costs = []
K_range = [3, 5, 8, 10, 12, 15, 20]

for k in K_range:
    kmeans_test = KMeans(featuresCol="features", k=k, seed=42)
    model_test = kmeans_test.fit(data_scaled)
    cost = model_test.summary.trainingCost
    costs.append(cost)
    print(f"K={k} -> Coût (inertie) = {cost:.2f}")


# 9. ENTRAINER LE MODELE FINAL AVEC LE K CHOISI

K_OPTIMAL = 10

kmeans = KMeans(featuresCol="features", k=K_OPTIMAL, seed=42)
model = kmeans.fit(data_scaled)

predictions = model.transform(data_scaled)
predictions.cache()

# 10. EVALUER LE MODELE - score de silhouette
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")
silhouette = evaluator.evaluate(predictions)
print(f"\n=== SCORE DE SILHOUETTE : {silhouette:.4f} ===")
print("(plus proche de 1 = clusters bien séparés, proche de 0 = clusters qui se chevauchent)")

# 11. ANALYSER LES CLUSTERS
print("\n=== TAILLE DE CHAQUE CLUSTER ===")
predictions.groupBy("prediction").count().orderBy("prediction").show()

print("\n=== EXEMPLES DE SERIES PAR CLUSTER (3 premiers clusters) ===")
for cluster_id in range(min(3, K_OPTIMAL)):
    print(f"\n--- Cluster {cluster_id} ---")
    predictions.filter(col("prediction") == cluster_id) \
        .select("name", "popularity", "vote_average") \
        .orderBy(col("popularity").desc()) \
        .limit(5) \
        .show(truncate=False)

# 12. SAUVEGARDER LE RESULTAT SUR HDFS
result = predictions.select(
    "show_id", "name", "popularity", "vote_average", "vote_count",
    "prediction"
).withColumnRenamed("prediction", "cluster_id")

result.write.mode("overwrite").parquet(f"{HDFS_PATH}/shows_clustered")

print("\n Clustering terminé et sauvegardé sur HDFS !")

spark.stop()