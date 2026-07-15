import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# ============================================================
# 1. PARAMÈTRES
# ============================================================

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

NUMBER_OF_CLUSTERS = 8
SEED = 42

OUTPUT_DIRECTORY = Path("outputs")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)


# ============================================================
# 2. DÉMARRAGE DE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("TV Series - Final KMeans")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 3. CHARGEMENT DES DONNÉES NETTOYÉES
# ============================================================

clean_data_path = f"{HDFS_PATH}/shows_clean"

df = spark.read.parquet(
    clean_data_path
)

number_of_rows = df.count()

print("Nombre de lignes au départ :", number_of_rows)

# Le modèle ne peut pas utiliser une décennie inconnue
df = df.filter(
    col("decade").isNotNull()
)


# ============================================================
# 4. UNE LIGNE PAR SÉRIE ET ENCODAGE DES GENRES
# ============================================================

shows_genres = df.groupBy(
    "show_id",
    "name",
    "popularity",
    "vote_average",
    "vote_count",
    "number_of_seasons",
    "number_of_episodes",
    "episode_run_time",
    "decade",
    "adult"
).pivot(
    "genre_name"
).agg(
    lit(1)
)

# Un genre absent prend la valeur 0
shows_genres = shows_genres.na.fill(0)

number_after_grouping = shows_genres.count()

print(
    "Nombre de séries après regroupement :",
    number_after_grouping
)


# ============================================================
# 5. FILTRAGE DES DONNÉES
# ============================================================

shows_genres = shows_genres.filter(
    col("episode_run_time") >= 0
)

shows_genres = shows_genres.filter(
    col("episode_run_time") < 180
)

shows_genres = shows_genres.filter(
    col("vote_count") > 0
)

shows_genres = shows_genres.filter(
    col("adult") == 0
)

shows_genres.cache()

number_of_series = shows_genres.count()

print(
    "Nombre de séries utilisées par le modèle :",
    number_of_series
)


# ============================================================
# 6. CRÉATION DE NOUVELLES FEATURES
# ============================================================

# Longueur simplifiée :
# 1 = une saison
# 2 = deux ou trois saisons
# 3 = quatre à sept saisons
# 4 = plus de sept saisons

shows_genres = shows_genres.withColumn(
    "serie_length",
    when(
        col("number_of_seasons") == 1,
        1
    ).when(
        col("number_of_seasons") <= 3,
        2
    ).when(
        col("number_of_seasons") <= 7,
        3
    ).otherwise(4)
)

# Note pondérée par le nombre de votes
shows_genres = shows_genres.withColumn(
    "vote_quality",
    col("vote_average")
    * (
        col("vote_count")
        / (col("vote_count") + 100)
    )
)

# Nombre moyen d'épisodes par saison
shows_genres = shows_genres.withColumn(
    "content_density",
    when(
        col("number_of_seasons") > 0,
        col("number_of_episodes")
        / col("number_of_seasons")
    ).otherwise(0)
)


# ============================================================
# 7. SÉLECTION DES FEATURES
# ============================================================

excluded_columns = [
    "show_id",
    "name",
    "vote_average",
    "number_of_seasons",
    "number_of_episodes",
    "episode_run_time",
    "adult",
    "null"
]

feature_columns = []

for column_name in shows_genres.columns:

    if column_name not in excluded_columns:
        feature_columns.append(column_name)

print("\nFeatures utilisées :")

for feature in feature_columns:
    print("-", feature)

print("\nNombre de features :", len(feature_columns))


# Sauvegarde de l'ordre des features
feature_file_path = OUTPUT_DIRECTORY / "feature_cols.json"

with open(
    feature_file_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_columns,
        file,
        indent=4
    )


# ============================================================
# 8. ASSEMBLAGE DES FEATURES
# ============================================================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features_raw",
    handleInvalid="skip"
)

data_assembled = assembler.transform(
    shows_genres
)


# ============================================================
# 9. NORMALISATION
# ============================================================

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(
    data_assembled
)

data_scaled = scaler_model.transform(
    data_assembled
)

data_scaled.cache()


# Sauvegarde du scaler
scaler_path = f"{HDFS_PATH}/models/scaler_model"

scaler_model.write().overwrite().save(
    scaler_path
)


# ============================================================
# 10. ENTRAÎNEMENT DU K-MEANS FINAL
# ============================================================

kmeans = KMeans(
    featuresCol="features",
    predictionCol="prediction",
    k=NUMBER_OF_CLUSTERS,
    seed=SEED
)

model = kmeans.fit(
    data_scaled
)

predictions = model.transform(
    data_scaled
)

predictions.cache()


# Sauvegarde du modèle K-Means
model_path = f"{HDFS_PATH}/models/kmeans_model"

model.write().overwrite().save(
    model_path
)


# ============================================================
# 11. ÉVALUATION DU MODÈLE
# ============================================================

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction"
)

silhouette = evaluator.evaluate(
    predictions
)

print("\nScore de silhouette :", round(silhouette, 4))


# Calcul de la taille de chaque cluster
cluster_size_rows = (
    predictions
    .groupBy("prediction")
    .count()
    .orderBy("prediction")
    .collect()
)

cluster_sizes = {}

print("\n=== TAILLE DES CLUSTERS ===")

for row in cluster_size_rows:

    cluster_id = int(
        row["prediction"]
    )

    cluster_count = int(
        row["count"]
    )

    cluster_sizes[str(cluster_id)] = cluster_count

    print(
        "Cluster",
        cluster_id,
        ":",
        cluster_count,
        "séries"
    )


# ============================================================
# 12. RAPPORT DE L'ENTRAÎNEMENT
# ============================================================

report = {
    "k": NUMBER_OF_CLUSTERS,
    "seed": SEED,
    "silhouette": round(silhouette, 4),
    "nb_series": predictions.count(),
    "nb_features": len(feature_columns),
    "cluster_sizes": cluster_sizes
}

report_path = OUTPUT_DIRECTORY / "model_report.json"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# 13. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

results = predictions.select(
    "show_id",
    "name",
    "popularity",
    "vote_quality",
    "vote_count",
    "decade",
    "serie_length",
    "content_density",
    "prediction"
)

results = results.withColumnRenamed(
    "prediction",
    "cluster_id"
)

clustered_data_path = f"{HDFS_PATH}/shows_clustered"

results.write.mode("overwrite").parquet(
    clustered_data_path
)


# ============================================================
# 14. FIN
# ============================================================

print("\nEntraînement terminé.")

print("\nFichiers sauvegardés :")
print("-", scaler_path)
print("-", model_path)
print("-", clustered_data_path)
print("-", feature_file_path)
print("-", report_path)

spark.stop()