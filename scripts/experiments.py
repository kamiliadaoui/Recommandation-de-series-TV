import os
import csv
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# ============================================================
# 1. DÉMARRAGE DE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("TV Series - Experiments")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/user/data"
OUTPUT_DIR = "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CHARGEMENT DES DONNÉES NETTOYÉES
# ============================================================

df = spark.read.parquet(
    f"{HDFS_PATH}/shows_clean"
)

# K-Means ne peut pas utiliser une décennie inconnue
df = df.filter(
    col("decade").isNotNull()
)


# ============================================================
# 3. UNE LIGNE PAR SÉRIE + ENCODAGE DES GENRES
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

# Les genres absents sont remplacés par 0
shows_genres = shows_genres.na.fill(0)


# ============================================================
# 4. FILTRAGE DES DONNÉES
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


# ============================================================
# 5. CRÉATION DE NOUVELLES FEATURES
# ============================================================

# Longueur simplifiée de la série :
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

# La note est pondérée par le nombre de votes
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

shows_genres.cache()

number_of_series = shows_genres.count()

print("\nNombre de séries utilisées :", number_of_series)


# ============================================================
# 6. CHOIX DES FEATURES
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


# ============================================================
# 7. ASSEMBLAGE DES FEATURES
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
# 8. NORMALISATION
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


# ============================================================
# 9. PRÉPARATION DE L'ÉVALUATION
# ============================================================

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction"
)

k_values = [3, 5, 8, 10, 12, 15]
seeds = [42, 123, 456]

experiments = []


# ============================================================
# 10. ENTRAÎNEMENT DES MODÈLES
# ============================================================

print("\n=== DÉBUT DES EXPÉRIMENTATIONS ===")

for k in k_values:

    for seed in seeds:

        print("\nEntraînement :")
        print("K =", k)
        print("Seed =", seed)

        kmeans = KMeans(
            featuresCol="features",
            k=k,
            seed=seed
        )

        model = kmeans.fit(
            data_scaled
        )

        predictions = model.transform(
            data_scaled
        )

        # Coût du modèle
        cost = model.summary.trainingCost

        # Qualité de séparation des clusters
        silhouette = evaluator.evaluate(
            predictions
        )

        # Taille de chaque cluster
        cluster_rows = (
            predictions
            .groupBy("prediction")
            .count()
            .collect()
        )

        cluster_sizes = []

        for row in cluster_rows:
            cluster_sizes.append(
                row["count"]
            )

        minimum_cluster_size = min(
            cluster_sizes
        )

        maximum_cluster_size = max(
            cluster_sizes
        )

        print("Coût :", round(cost, 2))
        print("Silhouette :", round(silhouette, 4))
        print("Plus petit cluster :", minimum_cluster_size)
        print("Plus grand cluster :", maximum_cluster_size)

        result = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "k": k,
            "seed": seed,
            "cout": round(cost, 2),
            "silhouette": round(silhouette, 4),
            "nb_series": number_of_series,
            "nb_features": len(feature_columns),
            "min_cluster_size": minimum_cluster_size,
            "max_cluster_size": maximum_cluster_size
        }

        experiments.append(result)


# ============================================================
# 11. SAUVEGARDE DES RÉSULTATS
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "results.csv"
)

column_names = [
    "date",
    "k",
    "seed",
    "cout",
    "silhouette",
    "nb_series",
    "nb_features",
    "min_cluster_size",
    "max_cluster_size"
]

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=column_names
    )

    writer.writeheader()
    writer.writerows(experiments)

print("\nRésultats sauvegardés dans :", output_csv)


# ============================================================
# 12. CHOIX DU MEILLEUR MODÈLE
# ============================================================

# On retire les modèles contenant un cluster trop petit
valid_experiments = []

maximum_allowed_size = number_of_series * 0.70

for result in experiments:

    cluster_is_not_too_small = (
        result["min_cluster_size"] >= 100
    )

    cluster_is_not_too_large = (
        result["max_cluster_size"] <= maximum_allowed_size
    )

    if cluster_is_not_too_small and cluster_is_not_too_large:
        valid_experiments.append(result)

# Si aucun modèle ne respecte la condition,
# on utilise tous les résultats
if len(valid_experiments) == 0:
    valid_experiments = experiments

# Recherche manuelle de la meilleure silhouette
best_model = valid_experiments[0]

for result in valid_experiments:

    if result["silhouette"] > best_model["silhouette"]:
        best_model = result

print("\n=== MEILLEUR MODÈLE VALIDE ===")
print("K :", best_model["k"])
print("Seed :", best_model["seed"])
print("Silhouette :", best_model["silhouette"])
print("Coût :", best_model["cout"])
print(
    "Plus petit cluster :",
    best_model["min_cluster_size"]
)
print(
    "Plus grand cluster :",
    best_model["max_cluster_size"]
)

spark.stop()